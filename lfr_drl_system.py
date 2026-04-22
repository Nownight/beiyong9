"""
LFR 动态瞄准策略优化：单文件实现

该文件将以下四个模块合并实现：
1) weather_parser.py
2) lfr_otsun_env.py
3) ppo_trainer.py
4) tonatiuh_exporter.py

说明：
- OTSun 接口在本文件中采用真实光线追迹（run_otsun_tracing）实现 DRL 物理闭环。
- 代码使用面向对象设计，并提供中文 Docstring 与类型注解。
"""

from __future__ import annotations

import sys
import os
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

FREECAD_BIN = r"D:\software\FreeCAD 1.1\bin"
sys.path.append(FREECAD_BIN)

import FreeCAD
import Part
print("FreeCAD 导入成功！版本:", FreeCAD.Version())

import otsun
import numpy as np
from multiprocessing import Pool

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import math

import pandas as pd

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO


# =========================
# 全局物理与几何硬性参数
# =========================
LATITUDE_DEG: float = 40.14
LONGITUDE_DEG: float = 94.68

NUM_MIRRORS: int = 18
MIRROR_LENGTH_M: float = 10.0
MIRROR_REFLECTIVITY: float = 0.94

RECEIVER_HEIGHT_M: float = 9.0

CPC_ACCEPT_HALF_ANGLE_DEG: float = 45.0
CPC_APERTURE_HALF_WIDTH_M: float = 0.167
CPC_APERTURE_FULL_WIDTH_M: float = 0.334

ABSORBER_OUTER_RADIUS_M: float = 0.032
GLASS_ENVELOPE_OUTER_RADIUS_M: float = 0.059

ACTION_LOW_M: float = -0.167
ACTION_HIGH_M: float = 0.167


@dataclass(frozen=True)
class RewardWeights:
    """奖励函数权重。

    物理含义：
    - w_eff：鼓励系统提高光学效率。
    - w_flux_std：惩罚集热管表面能流不均匀性（标准差越大越差）。
    - spill_coeff：惩罚溢出损失（超出 CPC 有效开口时的能量浪费）。
    """

    w_eff: float = 1.0
    w_flux_std: float = 0.15
    spill_coeff: float = 3.0


class WeatherParser:
    """敦煌 TMY 气象解析与太阳位置计算模块。

    该类负责：
    1) 读取 CSV（Year, Month, Day, Hour, DNI）；
    2) 过滤 DNI<=0 夜间无效数据；
    3) 使用 Cooper 简化公式估算太阳赤纬；
    4) 结合地理位置计算太阳高度角 alpha_s 和太阳方位角 gamma_s。

    约定：
    - alpha_s：太阳高度角，单位度。
    - gamma_s：太阳方位角，单位度。此处采用常见定义：
      以正南为 0°，向西为正，范围约 [-180°, 180°]。
    """

    required_columns = ["Year", "Month", "Day", "Hour", "DNI"]

    def __init__(self, csv_path: str | Path, latitude_deg: float = LATITUDE_DEG, longitude_deg: float = LONGITUDE_DEG) -> None:
        self.csv_path = Path(csv_path)
        self.latitude_deg = latitude_deg
        self.longitude_deg = longitude_deg
        self._df = self._load_and_clean()

    def _load_and_clean(self) -> pd.DataFrame:
        """加载并清洗 TMY 数据。"""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"未找到气象文件：{self.csv_path}")

        df = pd.read_csv(self.csv_path)
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"CSV 缺少必要列：{missing}，需要列：{self.required_columns}")

        for col in ["Year", "Month", "Day", "Hour"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        df["DNI"] = pd.to_numeric(df["DNI"], errors="coerce")

        df = df.dropna(subset=self.required_columns)
        df = df[df["DNI"] > 0].copy()
        if df.empty:
            raise ValueError("过滤 DNI<=0 后无可用白天数据，请检查输入 CSV。")

        df = df.reset_index(drop=True)
        return df

    @property
    def data(self) -> pd.DataFrame:
        """返回清洗后的白天有效数据。"""
        return self._df

    @staticmethod
    def _day_of_year(year: int, month: int, day: int) -> int:
        """计算积日 N（1-365/366）。"""
        return datetime(year, month, day).timetuple().tm_yday

    @staticmethod
    def _declination_cooper(day_of_year: int) -> float:
        """Cooper 简化公式计算太阳赤纬（弧度）。"""
        return math.radians(23.45) * math.sin(math.radians(360.0 * (284 + day_of_year) / 365.0))

    def get_solar_vector(self, index: int) -> Tuple[float, float, float]:
        """返回指定索引时刻太阳高度角、方位角与 DNI。

        Args:
            index: 清洗后数据的行索引。

        Returns:
            (alpha_s_deg, gamma_s_deg, dni)
        """
        if index < 0 or index >= len(self._df):
            raise IndexError(f"索引越界：index={index}, 数据长度={len(self._df)}")

        row = self._df.iloc[index]
        year, month, day = int(row["Year"]), int(row["Month"]), int(row["Day"])
        hour = int(row["Hour"])
        dni = float(row["DNI"])

        n = self._day_of_year(year, month, day)
        delta = self._declination_cooper(n)

        phi = math.radians(self.latitude_deg)
        # 简化处理：TMY 小时直接映射为地方太阳时整点中心（hour-12）
        omega = math.radians(15.0 * (hour - 12.0))

        sin_alpha = math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.cos(omega)
        sin_alpha = float(np.clip(sin_alpha, -1.0, 1.0))
        alpha = math.asin(sin_alpha)

        # 方位角：以正南为0，向西为正。
        cos_alpha = max(1e-8, math.cos(alpha))
        sin_gamma = (math.cos(delta) * math.sin(omega)) / cos_alpha
        cos_gamma = (math.sin(alpha) * math.sin(phi) - math.sin(delta)) / (cos_alpha * math.cos(phi) + 1e-8)
        gamma = math.atan2(sin_gamma, cos_gamma)

        return math.degrees(alpha), math.degrees(gamma), dni


class LFROTSunEnv(gym.Env):
    """LFR + OTSun 真实光线追迹强化学习环境。"""

    metadata = {"render_modes": []}

    def __init__(self, weather_parser: WeatherParser, reward_weights: Optional[RewardWeights] = None, max_episode_steps: int = 24) -> None:
        super().__init__()
        self.weather = weather_parser
        self.reward_weights = reward_weights or RewardWeights()
        self.max_episode_steps = max_episode_steps

        self.action_space = spaces.Box(
            low=np.full((NUM_MIRRORS,), ACTION_LOW_M, dtype=np.float32),
            high=np.full((NUM_MIRRORS,), ACTION_HIGH_M, dtype=np.float32),
            shape=(NUM_MIRRORS,),
            dtype=np.float32,
        )

        # alpha in [0, 90], gamma in [-180, 180], DNI in [0, 1200]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )

        self._current_idx: int = 0
        self._steps: int = 0
        self._rng = np.random.default_rng(seed=42)

        # -------- OTSun 缓存（性能关键）--------
        self._ots_scene: Any = None
        self._ots_light: Any = None
        self._ots_mirrors: List[Any] = []
        self._ots_absorber: Any = None
        self._ots_cpc: Any = None
        self._doc: Any = None
        self._sel: List[Any] = []
        self._tracking: Any = None
        self._emitting_region: Any = None
        self._spectrum_cdf: Any = None
        self._direction_distribution: Any = None
        self._main_direction: np.ndarray = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self._mirror_centers: np.ndarray = np.zeros((NUM_MIRRORS, 3), dtype=np.float64)
        self._receiver_point = np.array([0.0, 0.0, RECEIVER_HEIGHT_M], dtype=np.float64)
        self._ray_count = 5000

        self._init_otsun_scene()

    @staticmethod
    def _normalize_state(alpha_s: float, gamma_s: float, dni: float) -> np.ndarray:
        """将物理状态量归一化到神经网络友好的尺度。"""
        alpha_n = np.clip(alpha_s / 90.0, 0.0, 1.0)
        gamma_n = np.clip(gamma_s / 180.0, -1.0, 1.0)
        dni_n = np.clip(dni / 1200.0, 0.0, 1.0)
        return np.array([alpha_n, gamma_n, dni_n], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        alpha_s, gamma_s, dni = self.weather.get_solar_vector(self._current_idx)
        return self._normalize_state(alpha_s, gamma_s, dni)

    @staticmethod
    def _safe_norm(vec: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        n = float(np.linalg.norm(vec))
        if n <= eps:
            return np.zeros_like(vec)
        return vec / n

    def _denormalize_state(self, state: np.ndarray) -> Tuple[float, float, float]:
        alpha_s = float(np.clip(state[0], 0.0, 1.0) * 90.0)
        gamma_s = float(np.clip(state[1], -1.0, 1.0) * 180.0)
        dni = float(np.clip(state[2], 0.0, 1.0) * 1200.0)
        return alpha_s, gamma_s, dni

    @staticmethod
    def _call_by_names(obj: Any, names: List[str], *args: Any, **kwargs: Any) -> Any:
        for name in names:
            fn = getattr(obj, name, None)
            if callable(fn):
                return fn(*args, **kwargs)
        raise AttributeError(f"{obj!r} 缺少可调用接口: {names}")

    def _set_vector_like(self, obj: Any, candidate_names: List[str], vec: np.ndarray) -> None:
        v = np.asarray(vec, dtype=float)
        for name in candidate_names:
            if hasattr(obj, name):
                attr = getattr(obj, name)
                if callable(attr):
                    attr(v.tolist())
                else:
                    setattr(obj, name, v.tolist())
                return
        raise AttributeError(f"无法在 {obj!r} 上设置向量属性: {candidate_names}")

    def _init_otsun_scene(self) -> None:
        """一次性初始化 OTSun 场景（纯代码建模，静态几何只构建一次）。"""
        if self._ots_scene is not None:
            return

        otsun.ReflectorSpecularLayer("Mir1", 0.95)
        otsun.ReflectorSpecularLayer("Mir2", 0.91)
        otsun.AbsorberSimpleLayer("Abs", 0.95)
        otsun.TransparentSimpleLayer("Trans", 0.965)

        doc_name = "LFR_DRL_PureCode"
        if FreeCAD.getDocument(doc_name) is not None:
            FreeCAD.closeDocument(doc_name)
        self._doc = FreeCAD.newDocument(doc_name)

        # ---- Absorber（吸收管）----
        absorber = self._doc.addObject("Part::Feature", "Absorber")
        absorber.Label = "Absorber"
        absorber.Shape = Part.makeCylinder(
            ABSORBER_OUTER_RADIUS_M,
            MIRROR_LENGTH_M,
            FreeCAD.Vector(0.0, -MIRROR_LENGTH_M / 2.0, RECEIVER_HEIGHT_M),
            FreeCAD.Vector(0.0, 1.0, 0.0),
        )

        # ---- Glass Envelope（玻璃罩）----
        glass = self._doc.addObject("Part::Feature", "GlassEnvelope")
        glass.Label = "GlassEnvelope"
        glass.Shape = Part.makeCylinder(
            GLASS_ENVELOPE_OUTER_RADIUS_M,
            MIRROR_LENGTH_M,
            FreeCAD.Vector(0.0, -MIRROR_LENGTH_M / 2.0, RECEIVER_HEIGHT_M),
            FreeCAD.Vector(0.0, 1.0, 0.0),
        )

        # ---- CPC（简化为沿槽长方向的实体导光槽）----
        cpc = self._doc.addObject("Part::Feature", "CPC")
        cpc.Label = "CPC"
        cpc_height = 0.45
        cpc.Shape = Part.makeBox(
            CPC_APERTURE_FULL_WIDTH_M,
            MIRROR_LENGTH_M,
            cpc_height,
            FreeCAD.Vector(-CPC_APERTURE_HALF_WIDTH_M, -MIRROR_LENGTH_M / 2.0, RECEIVER_HEIGHT_M - cpc_height / 2.0),
            FreeCAD.Vector(1.0, 0.0, 0.0),
            FreeCAD.Vector(0.0, 0.0, 1.0),
        )

        # ---- Primary Mirrors（18 面镜子）----
        mirror_width = 0.5
        mirror_thickness = 0.005
        mirror_span = mirror_width * NUM_MIRRORS
        x_start = -mirror_span / 2.0 + mirror_width / 2.0

        mirrors: List[Any] = []
        for i in range(NUM_MIRRORS):
            mirror = self._doc.addObject("Part::Feature", f"Mirror_{i+1:02d}")
            mirror.Label = f"PrimaryMirror_{i+1:02d}"
            x_center = x_start + i * mirror_width
            mirror.Shape = Part.makeBox(
                mirror_width,
                MIRROR_LENGTH_M,
                mirror_thickness,
                FreeCAD.Vector(x_center - mirror_width / 2.0, -MIRROR_LENGTH_M / 2.0, 0.0),
                FreeCAD.Vector(1.0, 0.0, 0.0),
                FreeCAD.Vector(0.0, 0.0, 1.0),
            )
            mirrors.append(mirror)

        self._doc.recompute()

        self._sel = [absorber, glass, cpc, *mirrors]
        scene = otsun.Scene(self._sel)

        # 按用户要求保存关键对象
        self.scene = scene
        self.mirrors = mirrors
        self.light_source = None

        # 兼容现有成员命名
        self._ots_scene = scene
        self._ots_absorber = absorber
        self._ots_cpc = cpc
        self._ots_mirrors = mirrors

        for i, mirror in enumerate(self._ots_mirrors):
            bbox = mirror.Shape.BoundBox
            self._mirror_centers[i] = np.array([bbox.Center.x, bbox.Center.y, bbox.Center.z], dtype=np.float64)

        self._spectrum_cdf = otsun.cdf_from_pdf_file(os.path.join("data", "ASTMG173-direct.txt"))
        self._direction_distribution = otsun.buie_distribution(0.05)
        self._main_direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self._tracking = otsun.MultiTracking(self._main_direction, self._ots_scene)
        self._tracking.make_movements()
        self._emitting_region = otsun.SunWindow(self._ots_scene, self._main_direction)
        self._ots_light = otsun.LightSource(
            self._ots_scene,
            self._emitting_region,
            self._spectrum_cdf,
            1.0,
            self._direction_distribution,
        )
        self.light_source = self._ots_light


    def _sun_direction(self, alpha_s_deg: float, gamma_s_deg: float) -> np.ndarray:
        """将高度角/方位角转成从太阳指向地面的单位向量。"""
        alpha = math.radians(alpha_s_deg)
        gamma = math.radians(gamma_s_deg)
        # 以正南为 x+，向西为 y+ 的近似局部坐标
        sx = math.cos(alpha) * math.cos(gamma)
        sy = math.cos(alpha) * math.sin(gamma)
        sz = math.sin(alpha)
        sun_from_ground = np.array([sx, sy, sz], dtype=np.float64)
        # OTSun 光线方向使用“传播方向”，因此取反（太阳 -> 地面）
        return self._safe_norm(-sun_from_ground)

    def _update_mirror_kinematics(self, sun_dir: np.ndarray, action: np.ndarray) -> None:
        """根据 action 映射 18 面镜法向，满足“太阳入射 -> CPC 目标点反射”几何。"""
        action = np.asarray(action, dtype=np.float64).reshape(NUM_MIRRORS)
        action = np.clip(action, ACTION_LOW_M, ACTION_HIGH_M)

        for i, mirror in enumerate(self._ots_mirrors):
            center = self._mirror_centers[i]

            # 每面镜的瞄准点：在 CPC 开口中心基础上沿 x 方向偏移。
            target = self._receiver_point.copy()
            target[0] += action[i]

            # 入射方向（指向镜面）与出射方向（镜面 -> 目标）
            in_dir = self._safe_norm(-sun_dir)
            out_dir = self._safe_norm(target - center)

            # 反射定律：法向为入射与出射单位向量的角平分向量
            bisector = in_dir + out_dir
            if np.linalg.norm(bisector) < 1e-9:
                # 极端退化：入射与出射几乎完全相反，回退到朝向目标的“抬升法向”
                bisector = out_dir + np.array([0.0, 0.0, 1.0], dtype=np.float64)
            normal = self._safe_norm(bisector)

            # 尽量保证镜面朝上，避免法向翻转造成追迹异常
            if normal[2] < 0:
                normal = -normal

            # 中文注释：优先走 OTSun/对象自带 set_normal；若无，则回退为 FreeCAD Placement 旋转。
            if hasattr(mirror, "set_normal") and callable(mirror.set_normal):
                mirror.set_normal(normal.tolist())
            elif hasattr(mirror, "normal"):
                mirror.normal = normal.tolist()
            else:
                src = FreeCAD.Vector(0.0, 0.0, 1.0)
                dst = FreeCAD.Vector(float(normal[0]), float(normal[1]), float(normal[2]))
                rot = FreeCAD.Rotation(src, dst)
                mirror.Placement = FreeCAD.Placement(mirror.Placement.Base, rot)

        if hasattr(self._doc, "recompute"):
            self._doc.recompute()

    def _update_sun_source(self, sun_dir: np.ndarray, dni: float) -> None:
        """更新 OTSun 光源方向与 DNI。"""
        # 中文注释：每一步仅更新太阳方向，不重新实例化静态场景；追迹对象按需重建以减少几何构建开销。
        self._main_direction = np.asarray(sun_dir, dtype=np.float64)
        if self._tracking is not None:
            self._tracking.undo_movements()
        self._tracking = otsun.MultiTracking(self._main_direction, self._ots_scene)
        self._tracking.make_movements()
        self._emitting_region = otsun.SunWindow(self._ots_scene, self._main_direction)
        self._ots_light = otsun.LightSource(
            self._ots_scene,
            self._emitting_region,
            self._spectrum_cdf,
            max(1.0, float(dni) / 900.0),
            self._direction_distribution,
        )

    def _extract_tracing_metrics(self, tracing_result: Any, dni: float) -> Tuple[float, float, float]:
        """从 Experiment 提取效率、能流标准差和溢出比例。"""
        exp = tracing_result
        aperture_collector_Th = 11 * 0.5 * 32 * 1_000_000
        source_term = max(exp.number_of_rays / exp.light_source.emitting_region.aperture, 1e-9)
        optical_efficiency = float(np.clip((exp.captured_energy_Th / aperture_collector_Th) / source_term, 0.0, 1.0))

        # 兼容不同版本：若没有细粒度通量则退化为单值分布
        if hasattr(exp, "flux_in_elements"):
            flux_arr = np.asarray(exp.flux_in_elements, dtype=np.float64).reshape(-1)
        else:
            flux_arr = np.array([float(getattr(exp, "captured_energy_Th", 0.0))], dtype=np.float64)
        flux_mean = max(float(np.mean(flux_arr)), 1e-9)
        flux_std = float(np.std(flux_arr) / flux_mean)

        spilled = float(getattr(exp, "escaped_energy", 0.0))
        total = max(float(getattr(exp, "total_energy", spilled + getattr(exp, "captured_energy_Th", 0.0))), 1e-9)
        spill_penalty = float(np.clip(spilled / total, 0.0, 1.0))
        return optical_efficiency, flux_std, spill_penalty

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境。"""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._steps = 0
        self._current_idx = int(self._rng.integers(0, len(self.weather.data)))
        return self._get_obs(), {"data_index": self._current_idx}

    def run_otsun_tracing(self, state: np.ndarray, action: np.ndarray) -> Tuple[float, float, float]:
        """真实 OTSun 光线追迹：更新镜面与太阳，再执行追迹并提取指标。"""
        alpha_s, gamma_s, dni = self._denormalize_state(np.asarray(state, dtype=np.float32).reshape(3))
        action = np.asarray(action, dtype=np.float32).reshape(NUM_MIRRORS)

        sun_dir = self._sun_direction(alpha_s, gamma_s)

        # 1) 更新一次镜几何姿态（仅改法向，不重建场景）
        self._update_mirror_kinematics(sun_dir, action)

        # 2) 更新太阳方向与强度
        self._update_sun_source(sun_dir, dni)

        # 3) 执行追迹（2000~5000 条光线区间内取 5000）
        exp = otsun.Experiment(self._ots_scene, self._ots_light, int(self._ray_count), None)
        exp.run()

        # 4) 提取三项奖励指标
        return self._extract_tracing_metrics(exp, dni)

    def step(self, action: np.ndarray):
        """环境一步推进：执行动作并计算奖励。"""
        obs = self._get_obs()
        optical_eff, flux_std, spill_penalty = self.run_otsun_tracing(obs, action)

        rw = self.reward_weights
        reward = rw.w_eff * optical_eff - rw.w_flux_std * flux_std - rw.spill_coeff * spill_penalty

        self._steps += 1
        self._current_idx = (self._current_idx + 1) % len(self.weather.data)

        terminated = False
        truncated = self._steps >= self.max_episode_steps
        info = {
            "optical_efficiency": optical_eff,
            "flux_std": flux_std,
            "spill_penalty": spill_penalty,
            "reward_raw": reward,
        }

        next_obs = self._get_obs()
        return next_obs, float(reward), terminated, truncated, info


class PPOTrainer:
    """PPO 训练与推理封装。

    提供从环境构建、模型训练、保存、以及典型时刻推理的一体化流程。
    """

    def __init__(self, csv_path: str | Path, model_path: str | Path = "ppo_lfr_otsun.zip") -> None:
        self.csv_path = Path(csv_path)
        self.model_path = Path(model_path)
        self.weather = WeatherParser(self.csv_path)
        self.env = LFROTSunEnv(self.weather)

    def train(self, total_timesteps: int = 100_000, seed: int = 42) -> None:
        """执行 PPO 训练并保存模型。"""
        if PPO is None:
            raise ImportError("stable_baselines3 未安装，请先执行：pip install stable-baselines3")

        model = PPO(
            policy="MlpPolicy",
            env=self.env,
            verbose=1,
            seed=seed,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.0,
            clip_range=0.2,
        )

        model.learn(total_timesteps=total_timesteps)
        model.save(str(self.model_path))

    def _find_summer_solstice_noon_index(self) -> Optional[int]:
        """在数据中查找“夏至正午”近似时刻（6月21日12时优先）。"""
        df = self.weather.data
        cond = (df["Month"] == 6) & (df["Day"] == 21) & (df["Hour"] == 12)
        if cond.any():
            return int(df[cond].index[0])

        # 退化策略：选取 alpha_s 最大时刻
        best_idx = None
        best_alpha = -1e9
        for i in range(len(df)):
            alpha, _, _ = self.weather.get_solar_vector(i)
            if alpha > best_alpha:
                best_alpha = alpha
                best_idx = i
        return best_idx

    def predict_extreme_strategy(self) -> np.ndarray:
        """预测夏至正午（或高太阳高度角）时刻的最佳 18 镜偏移策略。"""
        if PPO is None:
            raise ImportError("stable_baselines3 未安装，请先执行：pip install stable-baselines3")
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在：{self.model_path}，请先训练。")

        model = PPO.load(str(self.model_path), env=self.env)
        idx = self._find_summer_solstice_noon_index()
        if idx is None:
            raise RuntimeError("未找到可用于推理的有效时刻。")

        alpha_s, gamma_s, dni = self.weather.get_solar_vector(idx)
        state = self.env._normalize_state(alpha_s, gamma_s, dni)
        action, _ = model.predict(state, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(NUM_MIRRORS)
        action = np.clip(action, ACTION_LOW_M, ACTION_HIGH_M)
        return action


class TonatiuhExporter:
    """Tonatiuh 验证参数生成器。

    将 DRL 输出的 18 镜瞄准偏移量，整理为可读报告与 JSON 文件，
    供用户手工或脚本导入 Tonatiuh 场景参数中进行高保真验证。
    """

    def __init__(self, output_path: str | Path = "tonatiuh_targeting_report.json") -> None:
        self.output_path = Path(output_path)

    def export(self, best_action: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """导出 Tonatiuh 参考参数。

        Args:
            best_action: 长度应为 18 的偏移数组（m）。
            meta: 附加元数据（如场景说明、时刻、训练配置等）。

        Returns:
            导出的字典对象。
        """
        arr = np.asarray(best_action, dtype=float).reshape(-1)
        if arr.size != NUM_MIRRORS:
            raise ValueError(f"best_action 长度必须为 {NUM_MIRRORS}，当前为 {arr.size}")

        arr = np.clip(arr, ACTION_LOW_M, ACTION_HIGH_M)

        mirrors: List[Dict[str, Any]] = []
        for i, offset in enumerate(arr, start=1):
            mirrors.append(
                {
                    "mirror_id": i,
                    "target_offset_x_m": float(offset),
                    "target_offset_y_m": 0.0,
                    "note": "相对 CPC 开口中心横向偏移",
                }
            )

        report: Dict[str, Any] = {
            "site": {"name": "Dunhuang", "lat_deg": LATITUDE_DEG, "lon_deg": LONGITUDE_DEG},
            "receiver": {
                "height_m": RECEIVER_HEIGHT_M,
                "cpc_aperture_half_width_m": CPC_APERTURE_HALF_WIDTH_M,
                "cpc_accept_half_angle_deg": CPC_ACCEPT_HALF_ANGLE_DEG,
                "absorber_outer_radius_m": ABSORBER_OUTER_RADIUS_M,
                "glass_envelope_outer_radius_m": GLASS_ENVELOPE_OUTER_RADIUS_M,
            },
            "field": {
                "num_mirrors": NUM_MIRRORS,
                "mirror_length_m": MIRROR_LENGTH_M,
                "mirror_reflectivity": MIRROR_REFLECTIVITY,
            },
            "recommended_targets": mirrors,
            "verification_hint": "请在 Tonatiuh 中输入上述 18 面镜偏移参数，并运行 1,000,000 rays 进行交叉验证。",
            "meta": meta or {},
        }

        self.output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    @staticmethod
    def print_human_readable(best_action: np.ndarray) -> None:
        """打印简洁可读的 Tonatiuh 参数日志。"""
        arr = np.asarray(best_action, dtype=float).reshape(-1)
        print("\n=== Tonatiuh 镜面瞄准参数（单位: m）===")
        for i, v in enumerate(arr, start=1):
            print(f"Mirror-{i:02d}: target_offset_x = {v:+.6f}, target_offset_y = +0.000000")
        print("建议：在 Tonatiuh 中设置 1,000,000 条光线进行验证。")


def run_end_to_end(
    csv_path: str | Path,
    total_timesteps: int = 100_000,
    model_path: str | Path = "ppo_lfr_otsun.zip",
    export_path: str | Path = "tonatiuh_targeting_report.json",
) -> np.ndarray:
    """一键执行端到端流程：训练 -> 推理 -> 导出。"""
    trainer = PPOTrainer(csv_path=csv_path, model_path=model_path)
    trainer.train(total_timesteps=total_timesteps)

    best_action = trainer.predict_extreme_strategy()
    print("\n夏至正午（或高太阳高度角）预测最佳 18 镜瞄准偏移：")
    print(best_action)

    exporter = TonatiuhExporter(output_path=export_path)
    exporter.export(
        best_action,
        meta={
            "scenario": "LFR dynamic aiming with PPO",
            "timesteps": total_timesteps,
        },
    )
    exporter.print_human_readable(best_action)
    return best_action


def _self_test() -> None:
    """快速自测：构造最小 CSV，验证数据流可跑通。"""
    tmp_csv = Path("_tmp_dunhuang_tmy.csv")
    df = pd.DataFrame(
        {
            "Year": [2020] * 8,
            "Month": [6] * 8,
            "Day": [21] * 8,
            "Hour": [8, 9, 10, 11, 12, 13, 14, 15],
            "DNI": [500, 650, 780, 860, 920, 880, 760, 600],
        }
    )
    df.to_csv(tmp_csv, index=False)

    weather = WeatherParser(tmp_csv)
    a, g, d = weather.get_solar_vector(0)
    assert d > 0 and -180 <= g <= 180 and -90 <= a <= 90

    env = LFROTSunEnv(weather)
    obs, _ = env.reset(seed=123)
    assert obs.shape == (3,)
    action = np.zeros((NUM_MIRRORS,), dtype=np.float32)
    nobs, reward, term, trunc, info = env.step(action)
    assert nobs.shape == (3,)
    assert isinstance(reward, float) and isinstance(info, dict)
    assert (not term) and isinstance(trunc, bool)

    tmp_csv.unlink(missing_ok=True)
    print("Self-test passed.")


if __name__ == "__main__":
    # 注释掉轻量自测
    # _self_test()
    
    # 取消注释，开始端到端完整训练
    run_end_to_end("dunhuang_tmy.csv", total_timesteps=100_000)
