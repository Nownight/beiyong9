"""
LFR 动态瞄准策略优化：单文件实现

该文件将以下四个模块合并实现：
1) weather_parser.py
2) lfr_otsun_env.py
3) ppo_trainer.py
4) tonatiuh_exporter.py

说明：
- OTSun 接口在本文件中以 Dummy 代理函数实现（run_otsun_tracing），用于打通 DRL 闭环。
- 代码使用面向对象设计，并提供中文 Docstring 与类型注解。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import math

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "未安装 gymnasium，请先执行：pip install gymnasium"
    ) from exc

try:
    from stable_baselines3 import PPO
except ImportError:  # pragma: no cover
    PPO = None  # type: ignore


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
    """LFR + OTSun Dummy 代理的强化学习环境。

    该环境将“动态瞄准策略”抽象为连续控制问题：
    - 状态（State）：[alpha_s, gamma_s, DNI] 的归一化连续向量；
    - 动作（Action）：18 面一次镜的横向瞄准偏移（m）；
    - 奖励（Reward）：效率提升 - 能流不均匀惩罚 - 溢出惩罚。

    注意：
    - run_otsun_tracing() 目前为简化代理模型，便于快速训练验证。
    - 后续可将此函数替换为真实 OTSun 光线追迹调用。
    """

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

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境。

        随机选取一个白天有效时刻作为 episode 起点。
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._steps = 0
        self._current_idx = int(self._rng.integers(0, len(self.weather.data)))
        return self._get_obs(), {"data_index": self._current_idx}

    def run_otsun_tracing(self, state: np.ndarray, action: np.ndarray) -> Tuple[float, float, float]:
        """Dummy OTSun 代理：用简化物理估算闭环指标。

        输入：
            state: 归一化状态 [alpha_n, gamma_n, dni_n]
            action: 18 镜面偏移量（m）

        返回：
            optical_efficiency: 光学效率（0~1）
            flux_std: 集热管能流标准差（归一化量纲）
            spill_penalty: 溢出惩罚项（>=0）

        简化思想：
        - 光学效率受太阳高度角（高太阳角更优）和瞄准偏移离散程度影响；
        - 偏移越接近 0，溢出通常越少；偏移绝对值越大，溢出风险增加；
        - 能流标准差近似由镜面偏移分布与太阳方位扰动共同决定。
        """
        alpha_n, gamma_n, dni_n = float(state[0]), float(state[1]), float(state[2])

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, ACTION_LOW_M, ACTION_HIGH_M)

        action_std = float(np.std(action))
        action_mean_abs = float(np.mean(np.abs(action)))

        sun_factor = 0.55 + 0.45 * alpha_n
        dni_factor = dni_n

        intercept_factor = float(np.exp(-8.0 * action_std))
        spill_penalty = max(0.0, (action_mean_abs / CPC_APERTURE_HALF_WIDTH_M) ** 1.8 - 0.2)

        optical_efficiency = MIRROR_REFLECTIVITY * sun_factor * dni_factor * intercept_factor
        optical_efficiency = float(np.clip(optical_efficiency, 0.0, 1.0))

        # gamma 扰动 + 偏移离散导致不均匀
        flux_std = float(0.08 + 1.4 * action_std + 0.12 * abs(gamma_n))

        return optical_efficiency, flux_std, float(spill_penalty)

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
    # 默认仅做轻量自测。完整训练请取消注释 run_end_to_end 并准备真实 dunhuang_tmy.csv。
    _self_test()
    # run_end_to_end("dunhuang_tmy.csv", total_timesteps=100_000)
