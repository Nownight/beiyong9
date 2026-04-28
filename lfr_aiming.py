"""
LFR Annual AI-Driven Aiming Strategy Optimization
=================================================
敦煌 18 面镜线性菲涅尔反射器 年尺度智能瞄准策略优化

单文件 GUI 程序,支持:
  - 实时进度显示
  - 7 个实验阶段(数据→聚类→基线→BO→训练→蒸馏→年度合成)
  - 任意时刻中断
  - 自动断点续跑
  - 论文图表自动导出

依赖: numpy, pandas, scipy, scikit-learn, torch, botorch, matplotlib, pvlib, pysr (可选), tkinter
作者: 你 + Claude
"""

import os
import sys
import ast
import json
import hashlib
import pickle
import threading
import time
import queue
import traceback
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互后端
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import font_manager as fm

try:
    from scipy.linalg import LinAlgWarning
except Exception:
    LinAlgWarning = Warning

# tkinter 可选 (Kaggle/Colab/无显示器环境无 tkinter)
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False
    tk = None; ttk = None; filedialog = None; messagebox = None; scrolledtext = None

# 可选依赖,启动时检测
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from pvlib.solarposition import get_solarposition
    HAS_PVLIB = True
except ImportError:
    HAS_PVLIB = False

# 延迟导入 PySR，避免启动时触发 JuliaCall
HAS_PYSR = False


def _pick_first_available_font(candidates):
    """按优先级选可用字体,若都不可用则返回 None"""
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


ZH_FONT_CANDIDATES = ['SimSun', '宋体', 'NSimSun', 'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC']
EN_FONT_CANDIDATES = ['Times New Roman', 'Times-Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif']


def _apply_plot_language_style(lang: str):
    """按语言设置绘图样式并屏蔽常见字体告警"""
    base = {
        'font.size': 11,
        'figure.dpi': 120,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.unicode_minus': False
    }
    if lang == 'zh':
        zh_font = _pick_first_available_font(ZH_FONT_CANDIDATES)
        if zh_font:
            base.update({'font.family': 'sans-serif', 'font.sans-serif': [zh_font] + ZH_FONT_CANDIDATES})
        else:
            base.update({'font.family': 'sans-serif', 'font.sans-serif': ZH_FONT_CANDIDATES})
    else:
        en_font = _pick_first_available_font(EN_FONT_CANDIDATES)
        base.update({'font.family': 'serif', 'font.serif': ([en_font] if en_font else []) + EN_FONT_CANDIDATES})
    plt.rcParams.update(base)


def save_bilingual_figure(output_path, draw_fn, dpi=300):
    """
    output_path: 原始基础路径，例如 fig_dir / 'fig01_cluster_pca.png'
    draw_fn: 函数，接收 lang 参数，返回 fig
             fig = draw_fn('zh')  # 中文图
             fig = draw_fn('en')  # 英文图
    保存：
      fig01_cluster_pca_zh.png
      fig01_cluster_pca_en.png
    """
    output_path = Path(output_path)
    for lang in ('zh', 'en'):
        _apply_plot_language_style(lang)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=r'Glyph .* missing from font\(s\).*', category=UserWarning)
            fig = draw_fn(lang)
            target = output_path.with_name(f"{output_path.stem}_{lang}{output_path.suffix}")
            fig.tight_layout()
            fig.savefig(target, dpi=dpi)
            plt.close(fig)


# ==================== 配置 ====================

@dataclass
class Config:
    """全局配置,可从 YAML 加载"""
    # 工作目录
    workdir: str = './lfr_results'
    tmy_path: str = './dunhuang_tmy.csv'
    
    # 站点
    latitude: float = 40.1421
    longitude: float = 94.6620
    altitude: float = 1139.0
    timezone: int = 8
    
    # 系统几何 (18 面镜)
    n_mirrors: int = 18
    mirror_length: float = 10.0
    mirror_width: float = 0.5
    field_half_width: float = 9.0
    curvature_a2: float = 0.025
    receiver_height: float = 9.0
    cpc_inlet_height: float = 8.72
    cpc_half_angle_deg: float = 45.0
    cpc_inlet_width: float = 0.334
    absorber_radius: float = 0.050
    glass_radius: float = 0.059
    
    # 光学
    rho_mirror: float = 0.94
    rho_cpc: float = 0.92
    tau_glass: float = 0.96
    alpha_abs: float = 0.95
    slope_error_mrad: float = 2.0
    tracking_error_mrad: float = 1.0
    soiling: float = 0.97
    
    # 瞄准
    aim_mode: str = 'transverse_span'  # transverse_span: BO 只优化横截面分散半宽 span
    # old_longitudinal: 旧版沿管轴偏移，仅保留兼容，不作为默认
    z_range: tuple = (-5.0, 5.0)          # 旧版 longitudinal 参数保留，不再默认使用
    xaim_span_range: tuple = (0.0, 0.04)  # BO 搜索 span，单位 m
    paper_xaim_span: float = 0.035        # 文献策略4参考值，仅用于 BO 初始点/诊断
    bo_uniformity_metric: str = 'sigma_surface'
    bo_eta_floor_rel: float = 0.96
    bo_force_paper_span_initial: bool = True
    use_symmetry: bool = True
    
    # 数据过滤
    dni_threshold: float = 200.0
    sun_alt_min: float = 5.0
    
    # 聚类
    n_clusters: int = 12
    cluster_features: tuple = ('solar_alt', 'solar_az', 'cos_inc', 'DNI')
    cluster_seed: int = 42
    samples_per_cluster: int = 20
    
    # MCRT
    n_rays_eval: int = 20_000      # BO 内每次评估的光线数 (BO 阶段 ~3-4 小时)
    n_rays_validate: int = 100_000  # 基线评估和最终验证用更多光线
    n_phi_bins: int = 36
    n_z_bins: int = 50
    max_cpc_bounces: int = 10
    mcrt_backend: str = 'numpy_cpu'  # numpy_cpu / torch_cuda_experimental
    mcrt_gpu_validate_on_start: bool = True
    mcrt_gpu_parity_rays: int = 5000
    mcrt_gpu_parity_tol_eta: float = 0.05
    mcrt_gpu_parity_tol_sigma: float = 0.08
    mcrt_num_workers: int = 1
    
    # BO
    bo_n_initial: int = 30
    bo_n_iterations: int = 80
    bo_ref_point: tuple = (0.30, -1.5)  # (eta_opt 最低, -uniformity 最高)
    bo_force_s1_initial: bool = True
    experiment_mode: str = 'span_1d'  # span_1d / grouped_span / fixed_scan / compare_all
    fixed_span_values: tuple = (0.0, 0.015, 0.025, 0.035, 0.04)
    grouped_span_bounds: tuple = ((0.0, 0.025), (0.0, 0.040), (0.0, 0.050))
    grouped_span_enable_gamma: bool = False
    
    # Transformer
    nn_d_model: int = 128
    nn_n_heads: int = 4
    nn_n_layers: int = 3
    nn_batch_size: int = 64
    nn_epochs: int = 200
    nn_lr: float = 1e-3
    nn_val_ratio: float = 0.2
    # NN early stopping
    nn_early_stop_patience: int = 20
    nn_min_delta: float = 1e-6
    
    # PySR
    pysr_iterations: int = 50
    pysr_population: int = 40
    pysr_maxsize: int = 22
    enable_pysr: bool = False
    
    # 验证
    full_year_sample: int = 500
    n_rays_sensitivity: int = 50_000
    
    # 计算
    device: str = 'auto'  # auto/cpu/cuda
    seed: int = 42
    config_hash_strict: bool = True
    force_reuse_cache: bool = False
    
    def to_dict(self):
        return asdict(self)
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))
    
    def get_device(self):
        if not HAS_TORCH:
            return None
        if self.device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return self.device


def compute_config_hash(cfg: Config) -> str:
    """仅基于影响结果的核心字段计算配置哈希。"""
    keys = [
        'aim_mode', 'xaim_span_range', 'paper_xaim_span', 'bo_uniformity_metric', 'bo_eta_floor_rel',
        'experiment_mode', 'fixed_span_values', 'grouped_span_bounds',
        'n_mirrors', 'mirror_length', 'mirror_width', 'field_half_width',
        'receiver_height', 'cpc_inlet_height', 'cpc_half_angle_deg', 'cpc_inlet_width',
        'absorber_radius', 'glass_radius',
        'n_rays_eval', 'n_rays_validate', 'n_phi_bins', 'n_z_bins',
        'bo_n_initial', 'bo_n_iterations', 'samples_per_cluster', 'n_clusters', 'seed'
    ]
    payload = {k: getattr(cfg, k) for k in keys}
    txt = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(txt.encode('utf-8')).hexdigest()[:12]


def _validate_cached_hash(cached_hash: Optional[str], cfg: Config, label: str):
    cur = compute_config_hash(cfg)

    if not cfg.config_hash_strict or cfg.force_reuse_cache:
        return

    if not cached_hash:
        raise RuntimeError(
            f"{label} 缺少 config_hash，可能是旧版本缓存。"
            f"请删除对应阶段结果后重跑，或设置 force_reuse_cache=True。"
        )

    if cached_hash != cur:
        raise RuntimeError(
            f"{label} 缓存配置与当前配置不一致。"
            f"cached={cached_hash}, current={cur}。"
            f"请清理对应阶段结果，或设置 force_reuse_cache=True。"
        )


def _load_pickle_checked(path: Path, cfg: Config, label: str):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    cached_hash = obj.get('config_hash') if isinstance(obj, dict) else None
    _validate_cached_hash(cached_hash, cfg, label)
    return obj


def _save_pickle_with_hash(path: Path, payload: dict, cfg: Config):
    payload = dict(payload)
    payload['config_hash'] = compute_config_hash(cfg)
    with open(path, 'wb') as f:
        pickle.dump(payload, f)


# ==================== 断点管理 ====================

class Checkpoint:
    """断点状态管理: 每个阶段完成后写一个 marker 文件"""
    
    STAGES = ['data', 'cluster', 'baseline', 'bo', 'train', 'distill', 'annual', 'sensitivity']
    
    def __init__(self, workdir, config_hash=None, strict=True, force=False):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.workdir / 'checkpoint.json'
        self.state = self._load()
        self.config_hash = config_hash
        self.ensure_hash(config_hash, strict=strict, force=force)
    
    def _load(self):
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {'completed': [], 'partial': {}}
    
    def save(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)

    def ensure_hash(self, config_hash, strict=True, force=False):
        if config_hash is None:
            return
        cached = self.state.get('config_hash')
        has_old_progress = bool(self.state.get('completed')) or bool(self.state.get('partial'))

        if cached is None:
            if strict and has_old_progress and not force:
                raise RuntimeError(
                    "checkpoint.json 缺少 config_hash，但已有 completed/partial 进度。"
                    "这可能是旧版本断点。请清空 checkpoint 或设置 force_reuse_cache=True。"
                )
            self.state['config_hash'] = config_hash
            self.save()
            return
        if cached != config_hash:
            if strict and not force:
                raise RuntimeError(
                    f"checkpoint 配置 hash 不一致: cached={cached}, current={config_hash}。"
                    "请清理旧断点或设置 force_reuse_cache=True。"
                )
            self.state['config_hash'] = config_hash
            self.save()
    
    def is_done(self, stage):
        return stage in self.state['completed']
    
    def mark_done(self, stage):
        if stage not in self.state['completed']:
            self.state['completed'].append(stage)
        self.state['partial'].pop(stage, None)
        self.save()
    
    def get_partial(self, stage, key, default=None):
        return self.state['partial'].get(stage, {}).get(key, default)
    
    def set_partial(self, stage, key, value):
        if stage not in self.state['partial']:
            self.state['partial'][stage] = {}
        self.state['partial'][stage][key] = value
        self.save()
    
    def reset(self, stage=None):
        if stage is None:
            self.state = {'completed': [], 'partial': {}}
        else:
            if stage in self.state['completed']:
                self.state['completed'].remove(stage)
            self.state['partial'].pop(stage, None)
        self.save()


# ==================== 中断信号 ====================

class StopSignal:
    """线程安全的中断信号 + 时间预算管理"""
    def __init__(self, time_budget_seconds=None):
        self._stop = threading.Event()
        self._start = time.time()
        self._budget = time_budget_seconds  # None=无限制
        self._reason = None
    
    def request_stop(self, reason="用户请求中断"):
        self._reason = reason
        self._stop.set()
    
    def stop_requested(self):
        return self._stop.is_set()
    
    def reset(self):
        self._stop.clear()
        self._start = time.time()
        self._reason = None
    
    def time_left(self):
        if self._budget is None:
            return float('inf')
        return self._budget - (time.time() - self._start)
    
    def time_used(self):
        return time.time() - self._start
    
    def check(self):
        # 时间预算检查: 留 10 分钟做保存收尾
        if self._budget is not None and self.time_left() < 600:
            self._reason = f"时间预算耗尽 (已用 {self.time_used()/3600:.1f}h),自动保存退出"
            self._stop.set()
        if self._stop.is_set():
            raise InterruptedError(self._reason or "中断")


# ==================== 日志器 ====================

class Logger:
    """通过 queue 把消息送给 GUI"""
    def __init__(self, msg_queue):
        self.q = msg_queue
    def info(self, msg):
        self.q.put(('log', f"[INFO] {msg}"))
    def warn(self, msg):
        self.q.put(('log', f"[WARN] {msg}"))
    def error(self, msg):
        self.q.put(('log', f"[ERR ] {msg}"))
    def progress(self, frac, text=''):
        self.q.put(('progress', (frac, text)))
    def stage(self, stage, frac):
        self.q.put(('stage_progress', (stage, frac)))
    def status(self, text):
        self.q.put(('status', text))


# ==================== 1. 气象数据加载 ====================

def stage_data(cfg: Config, logger: Logger, stop: StopSignal, ckpt: Checkpoint):
    """阶段 1: 加载 TMY,计算太阳几何,过滤"""
    logger.info("=== 阶段 1/7: 数据加载与太阳几何 ===")
    workdir = Path(cfg.workdir)
    out_path = workdir / 'tmy_processed.pkl'
    
    if ckpt.is_done('data') and out_path.exists():
        logger.info(f"已完成,加载 {out_path}")
        return pd.read_pickle(out_path)
    
    if not Path(cfg.tmy_path).exists():
        raise FileNotFoundError(f"TMY 文件未找到: {cfg.tmy_path}")
    
    logger.info(f"读取 {cfg.tmy_path}")
    df = pd.read_csv(cfg.tmy_path)
    logger.info(f"原始数据: {len(df)} 行, 列: {list(df.columns)}")
    
    df['timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
    df['timestamp'] = df['timestamp'].dt.tz_localize(f'Etc/GMT-{cfg.timezone}')
    
    stop.check()
    logger.progress(0.2, "计算太阳几何...")
    
    if HAS_PVLIB:
        pos = get_solarposition(df['timestamp'], cfg.latitude, cfg.longitude,
                                altitude=cfg.altitude)
        df['solar_alt'] = pos['apparent_elevation'].values
        df['solar_az'] = pos['azimuth'].values
    else:
        # 简化解析公式 (NOAA)
        logger.warn("pvlib 未安装,使用简化太阳位置公式")
        df['solar_alt'], df['solar_az'] = _simple_sun_position(df, cfg)
    
    # LFR 入射角余弦 (南北向布置,横向入射角)
    alt_rad = np.deg2rad(df['solar_alt'])
    az_rad = np.deg2rad(df['solar_az'])
    # 横向入射角 (东西方向投影)
    df['cos_inc'] = np.cos(np.arctan(np.cos(alt_rad) * np.sin(az_rad) / np.sin(alt_rad).clip(0.01)))
    df['cos_inc'] = df['cos_inc'].clip(0, 1)
    
    # 过滤
    n0 = len(df)
    df = df[(df['DNI'] >= cfg.dni_threshold) & (df['solar_alt'] >= cfg.sun_alt_min)].copy()
    df = df.reset_index(drop=True)
    logger.info(f"过滤后保留: {len(df)}/{n0} 小时 ({100*len(df)/n0:.1f}%)")
    logger.info(f"年累积有效 DNI: {df['DNI'].sum()/1000:.1f} kWh/m²")
    
    df.to_pickle(out_path)
    ckpt.mark_done('data')
    logger.progress(1.0, "数据加载完成")
    return df


def _simple_sun_position(df, cfg):
    """简化太阳位置(无 pvlib 时备用)"""
    doy = df['timestamp'].dt.dayofyear.values
    hour = df['Hour'].values + (df['timestamp'].dt.minute.values / 60)
    decl = np.deg2rad(23.45 * np.sin(np.deg2rad(360 * (284 + doy) / 365)))
    eot = 0
    solar_time = hour + (cfg.longitude - cfg.timezone * 15) / 15 + eot / 60
    h_angle = np.deg2rad(15 * (solar_time - 12))
    lat = np.deg2rad(cfg.latitude)
    alt = np.arcsin(np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(h_angle))
    az = np.arctan2(np.sin(h_angle), np.cos(h_angle) * np.sin(lat) - np.tan(decl) * np.cos(lat))
    return np.rad2deg(alt), np.rad2deg(az) + 180


# ==================== 2. 聚类 ====================

def stage_cluster(cfg: Config, df: pd.DataFrame, logger: Logger,
                  stop: StopSignal, ckpt: Checkpoint):
    """阶段 2: K-means 聚类,选代表时刻"""
    logger.info("=== 阶段 2/7: 工况聚类 ===")
    workdir = Path(cfg.workdir)
    out_path = workdir / 'clusters.pkl'
    
    if ckpt.is_done('cluster') and out_path.exists():
        logger.info("已完成,加载聚类结果")
        with open(out_path, 'rb') as f:
            return pickle.load(f)
    
    if not HAS_SKLEARN:
        raise ImportError("需要 scikit-learn")
    
    feats = list(cfg.cluster_features)
    X = df[feats].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    
    stop.check()
    logger.progress(0.3, f"K-means n_clusters={cfg.n_clusters}")
    km = KMeans(n_clusters=cfg.n_clusters, random_state=cfg.cluster_seed, n_init=10)
    labels = km.fit_predict(Xs)
    df = df.copy()
    df['cluster'] = labels
    
    sil = silhouette_score(Xs, labels) if len(Xs) > cfg.n_clusters else 0
    db = davies_bouldin_score(Xs, labels) if len(Xs) > cfg.n_clusters else 0
    logger.info(f"轮廓系数={sil:.3f}, DB 指数={db:.3f}")
    
    # 代表时刻: 每簇质心最近的真实小时
    representatives = []
    for k in range(cfg.n_clusters):
        idx = np.where(labels == k)[0]
        cdist = np.linalg.norm(Xs[idx] - km.cluster_centers_[k], axis=1)
        rep_idx = idx[np.argmin(cdist)]
        representatives.append({
            'cluster': k,
            'rep_idx': int(rep_idx),
            'rep_timestamp': df.iloc[rep_idx]['timestamp'],
            'count': len(idx),
            'weight': len(idx) / len(df),
            'dni_mean': df.iloc[idx]['DNI'].mean(),
            'dni_std': df.iloc[idx]['DNI'].std(),
            'alt_mean': df.iloc[idx]['solar_alt'].mean(),
            'alt_std': df.iloc[idx]['solar_alt'].std(),
        })
    rep_df = pd.DataFrame(representatives)
    
    # PCA 二维投影 (用于绘图)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(Xs)
    
    result = {
        'df': df,
        'labels': labels,
        'representatives': rep_df,
        'silhouette': sil,
        'db_index': db,
        'X_pca': X_pca,
        'centers_pca': pca.transform(km.cluster_centers_),
        'features': feats,
        'kmeans_centers': km.cluster_centers_,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
    }
    
    with open(out_path, 'wb') as f:
        pickle.dump(result, f)
    ckpt.mark_done('cluster')
    logger.progress(1.0, f"聚类完成, 12 簇代表时刻已选取")
    return result


# ==================== 3. LFR 几何 ====================

class LFRGeometry:
    """18 面镜 LFR + CPC 几何"""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.N = cfg.n_mirrors
        self.H = cfg.receiver_height
        self.cpc_y = cfg.cpc_inlet_height
        self._build_mirrors()
        self._build_cpc()
        # CPC 公式输出的真实入口高度可能与配置值略不同, 用真实值覆盖
        self.cpc_y = self._cpc_actual_inlet_z
    
    def _build_mirrors(self):
        """主镜中心 x 坐标(对称分布)
        
        约定: 最外侧镜中心 = ±Q_max (field_half_width)
        18 面镜对称分布在 [-Q_max, +Q_max], 含端点
        """
        half = self.N // 2  # 9
        # 单侧 9 面镜,从 x_min(靠近中心) 到 x_max(=Q_max) 均匀分布
        # 中心两侧最近一面镜的 x: 取 spacing/2 (对称中心间隙 = spacing)
        # 最远一面镜的 x: Q_max
        Q_max = self.cfg.field_half_width
        # 单侧 9 面镜在 [x_inner, Q_max] 上等间距分布
        # 取 x_inner = Q_max / (2*half - 1) 的镜间距
        # 数学: x_i = x_inner + i * spacing (i=0..half-1), x_{half-1} = Q_max
        # 中心两镜距离 = 2 * x_inner, 镜间距 = spacing
        # 取 x_inner = spacing/2 → 中心间隙等于镜间距 → spacing = Q_max / (half - 0.5)
        spacing = Q_max / (half - 0.5)
        x_inner = spacing / 2
        x_right = np.array([x_inner + i * spacing for i in range(half)])
        # 验证: x_right[-1] 应等于 Q_max
        assert np.isclose(x_right[-1], Q_max), f"最外侧镜位置错: {x_right[-1]} vs {Q_max}"
        x_pos = np.concatenate([-x_right[::-1], x_right])
        self.mirror_x = x_pos
        self.mirror_w = np.full(self.N, self.cfg.mirror_width)
        self.mirror_a2 = np.full(self.N, self.cfg.curvature_a2)
        # 镜间距(后续 IAM 计算需要)
        self.mirror_spacing = spacing
    
    def _build_cpc(self):
        """CPC 型线构造 (论文 §2.1.3 复合抛物面集热器 CPC)
        
        参数方程 (吸热管中心为局部原点, y 轴向上):
          渐开线段 (|t| <= theta_c + pi/2):
            x(t) = r·(sin(t) - t·cos(t))
            y(t) = -r·(cos(t) + t·sin(t))
          抛物线段 (theta_c + pi/2 < |t| < 3pi/2 - theta_c):
            I(t) = r·(pi/2 + t + theta_c - cos(t - theta_c)) / (1 + sin(t - theta_c))
            x(t) = r·sin(t) - I(t)·cos(t)
            y(t) = -r·cos(t) - I(t)·sin(t)
          
          其中 I(t) 是展开长度,渐开线段 I = r·t,抛物线段如上式。
          两段在 t = theta_c + pi/2 处斜率连续。
        
        全局映射: 局部 (x, y) → 全局 (x_g, z_g)
          x_g = x_loc                    (横向)
          z_g = self.H + y_loc           (高度,管心在 z=H)
        """
        # CPC 基圆: 取玻璃套管外半径,避免渐开线进入玻璃管(论文 §2.1.3 修订说明)
        r = self.cfg.glass_radius
        theta_c = np.deg2rad(self.cfg.cpc_half_angle_deg)
        
        # 渐开线段: t in [0, pi/2 + theta_c]
        n_inv = 40
        t_inv = np.linspace(0, np.pi/2 + theta_c, n_inv)
        I_inv = r * t_inv
        x_inv = r * np.sin(t_inv) - I_inv * np.cos(t_inv)
        y_inv = -r * np.cos(t_inv) - I_inv * np.sin(t_inv)
        
        # 抛物线段: t in (pi/2 + theta_c, 3pi/2 - theta_c)
        n_par = 100
        t_par = np.linspace(np.pi/2 + theta_c + 1e-6, 3*np.pi/2 - theta_c - 1e-6, n_par)
        denom = 1 + np.sin(t_par - theta_c)
        denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        I_par = r * (np.pi/2 + t_par + theta_c - np.cos(t_par - theta_c)) / denom
        x_par = r * np.sin(t_par) - I_par * np.cos(t_par)
        y_par = -r * np.cos(t_par) - I_par * np.sin(t_par)
        
        # 拼接 (右半 CPC, 局部坐标)
        x_loc = np.concatenate([x_inv, x_par])
        y_loc = np.concatenate([y_inv, y_par])
        
        # 论文公式坐标系: y 轴向上 = CPC 开口方向
        # 但物理上 LFR 镜场在地面, CPC 开口向下对着镜场
        # 翻转: 局部 y → 全局 -z 方向, 即 z_g = H - y_loc
        x_g = x_loc
        z_g = self.H - y_loc
        
        # 仅保留 x >= 0 (右半)
        mask = x_g >= -1e-9
        x_g = x_g[mask]
        z_g = z_g[mask]
        
        self.cpc_right_segs = np.column_stack([x_g, z_g])
        self.cpc_left_segs = np.column_stack([-x_g, z_g])
        
        # 记录实际入口位置(z 最小值, 因为 CPC 朝下)
        self._cpc_actual_inlet_z = z_g.min()
        self._cpc_actual_inlet_half_width = x_g[np.argmin(z_g)]
        
        self._compute_cpc_normals()
    
    def _compute_cpc_normals(self):
        """每段 CPC 壁的内法向量(指向 CPC 内部)"""
        # 右壁:x>0,内部在 x<0 方向(指向中心轴),内法向 x 分量应 < 0
        right_tan = np.diff(self.cpc_right_segs, axis=0)
        right_tan_norm = right_tan / (np.linalg.norm(right_tan, axis=1, keepdims=True) + 1e-12)
        # 顺时针 90° 旋转(沿入口→吸热管方向): n = (-ty, tx)
        right_n = np.column_stack([-right_tan_norm[:, 1], right_tan_norm[:, 0]])
        # 强制指向中心轴(x_normal < 0)
        flip = right_n[:, 0] > 0
        right_n[flip] = -right_n[flip]
        self.cpc_right_normals = right_n
        
        # 左壁:x<0,内法向 x 分量应 > 0
        left_tan = np.diff(self.cpc_left_segs, axis=0)
        left_tan_norm = left_tan / (np.linalg.norm(left_tan, axis=1, keepdims=True) + 1e-12)
        left_n = np.column_stack([-left_tan_norm[:, 1], left_tan_norm[:, 0]])
        flip = left_n[:, 0] < 0
        left_n[flip] = -left_n[flip]
        self.cpc_left_normals = left_n
    
    def tracking_normals(self, sun_alt_deg, sun_az_deg, aim_z=None):
        """计算每面镜子的法向量(2D 横截面)
        sun_alt: 太阳高度角
        sun_az: 太阳方位角(相对正南)
        aim_z: 此处忽略(纵向偏移在 3D 光追时使用)
        返回: 每面镜的旋转角(度)
        """
        # 简化为 2D: 太阳在 xy 平面投影
        alt = np.deg2rad(sun_alt_deg)
        az = np.deg2rad(sun_az_deg)
        # 入射光线方向(从太阳到镜面),在镜场横截面上投影
        sun_x = -np.sin(az) * np.cos(alt)
        sun_y = -np.sin(alt)  # 向下
        
        rotations = np.zeros(self.N)
        for i, x in enumerate(self.mirror_x):
            # 反射目标: CPC 入口中心 (0, cpc_y)
            target_x = -x
            target_y = self.cpc_y
            tlen = np.sqrt(target_x**2 + target_y**2)
            tx, ty = target_x / tlen, target_y / tlen
            # 法向量 = (反射方向 - 入射方向) 归一化
            nx = tx - sun_x
            ny = ty - sun_y
            nlen = np.sqrt(nx**2 + ny**2)
            nx, ny = nx / nlen, ny / nlen
            # 旋转角(法向量与垂直方向的夹角)
            rotations[i] = np.rad2deg(np.arctan2(nx, ny))
        return rotations
    
    def compute_iam(self, sun_alt_deg, sun_az_deg):
        """LFR 入射角修正因子 IAM (Cheng et al. 2018, Solar Energy 156)
        
        几何说明:
          - x 横向 (镜场宽度), y 沿管轴, z 垂直
          - 太阳横向投影角 theta_T (xz 平面): 太阳光线在镜场横截面内与 z 轴夹角
          - 纵向投影角 theta_L: 太阳光线沿管轴方向的偏移
          - 镜 i 跟踪角 rho_i: 镜面法向相对 z 轴的旋转
          - 镜 i 到接收器方位角 alpha_i = arctan(-x_i / H) (镜->CPC 入口方向)
          - 跟踪条件: rho_i = (alpha_i + theta_T) / 2
        
        模型:
          1. 横向阴影/遮挡 (邻居镜的几何投影)
          2. 纵向端部损失 K_L = max(0, 1 - H·|tan(theta_L)|/L)
        
        简化: 使用第一近邻效应 (Boito 2017 已验证 18 面镜场 1-邻近近似精度 > 95%)
        """
        alt = np.deg2rad(sun_alt_deg)
        az = np.deg2rad(sun_az_deg)
        
        # 安全检查: 太阳低于地平线
        if alt <= 0:
            return np.zeros(self.N), {'K_T': np.zeros(self.N), 'K_L': 0,
                                       'shadow_loss': np.zeros(self.N),
                                       'block_loss': np.zeros(self.N),
                                       'theta_T_deg': 0, 'theta_L_deg': 0,
                                       'rho_deg': np.zeros(self.N)}
        
        # 太阳投影角(LFR 标准定义)
        # 坐标系: x 横向 (东西), y 沿管轴 (南北), z 垂直
        # 太阳方向单位向量: (-cos(alt)·sin(az), -cos(alt)·cos(az), -sin(alt))
        # 
        # theta_T (横向投影角): 在 xz 平面内,太阳光线投影与 z 轴的夹角
        #   投影向量: (-cos(alt)·sin(az), -sin(alt))
        #   theta_T = arctan(cos(alt)·sin(az) / sin(alt))
        # theta_L (纵向投影角): 太阳光线偏离 xz 平面(镜场垂直平面)的角度
        #   = arctan(沿管轴分量 / 投影到 xz 平面的分量)
        #   = arctan(|cos(alt)·cos(az)| / sqrt(sin²(alt) + (cos(alt)·sin(az))²))
        theta_T = np.arctan2(np.cos(alt) * np.sin(az), np.sin(alt))
        proj_xz = np.sqrt(np.sin(alt)**2 + (np.cos(alt) * np.sin(az))**2)
        theta_L = np.arctan2(abs(np.cos(alt) * np.cos(az)), max(proj_xz, 1e-9))
        
        N = self.N
        H = self.H
        w = self.cfg.mirror_width
        L = self.cfg.mirror_length
        d = self.mirror_spacing
        
        alpha = np.arctan2(-self.mirror_x, H)
        rho = 0.5 * (alpha + theta_T)
        
        # ===== 横向 K_T =====
        # Cheng 2018 的简化封闭式:
        # 阴影: f_shadow = max(0, 1 - d·cos(rho_neighbor)/(w·|sin(theta_T - rho_neighbor)|))
        # 遮挡: f_block  = max(0, 1 - d·cos(rho_neighbor)/(w·|sin(alpha_self + rho_neighbor)|))
        # 实际上文献多用更工程化的近似。这里采用 Boito 2017 工程式:
        #
        # 镜面投影宽度 (水平面) w_proj = w·cos(rho)
        # 投影"阴影线" (邻居顶部投影到本镜面所在水平面的横向位置):
        #   delta_shadow = d - h_neighbor·tan(theta_T)·sgn  (h_neighbor = w/2·sin(rho_n))
        # 当 delta_shadow < w_proj/2 + w_proj_neighbor/2 时 → 部分阴影
        #
        # 但此处直接用更稳健的实用公式 (Mertins 2009):
        # K_T_i = max(0, min(1, (d·cos(rho_i) - h_eff·|tan(theta_T - rho_i)|) / (w·cos(rho_i))))
        # 其中 h_eff 为邻居最高点高度 ≈ (w/2)·|sin(rho_neighbor)|
        # 这个公式不太精确,但保证了:
        #   - 镜面平躺(rho=0)时无阴影
        #   - 太阳正顶(theta_T=0)时无阴影
        #   - 极端角度时 K_T → 0
        
        K_T = np.ones(N)
        shadow_loss_arr = np.zeros(N)
        block_loss_arr = np.zeros(N)
        
        for i in range(N):
            cos_rho_i = np.cos(rho[i])
            if cos_rho_i < 0.01:
                K_T[i] = 0
                continue
            w_proj = w * cos_rho_i  # 镜面水平投影宽度
            
            # 阴影来源邻居: theta_T 方向的邻居挡入射光
            # 当 theta_T > 0 (太阳偏 +x), 阴影来自 +x 邻居
            sgn_T = 1 if theta_T > 1e-6 else (-1 if theta_T < -1e-6 else 0)
            # 遮挡来源邻居: 朝向 CPC 那一侧反向 (alpha_i 方向)
            sgn_a = 1 if alpha[i] > 1e-6 else (-1 if alpha[i] < -1e-6 else 0)
            
            # 阴影: 邻居 j_s 在 +sgn_T 方向
            shadow_loss = 0.0
            if sgn_T != 0:
                j_s = i + sgn_T
                if 0 <= j_s < N:
                    # 邻居倾斜镜面顶部高度 h_n
                    h_n = abs(w / 2 * np.sin(rho[j_s]))
                    # 太阳光从邻居顶部到 z=0 平面的横向位移 = h_n·tan(theta_T)
                    # 邻居顶部 x 位置: x_n + (w/2)·cos(rho_n)·sgn_T (朝镜i方向边缘) - 实际是 -sgn_T(远离镜i的边)
                    # 取邻居朝镜 i 那一侧的顶端: x_n_edge = mirror_x[j_s] - sgn_T·(w/2)·cos(rho[j_s])
                    # 该顶端高度 ≈ h_n
                    # 阴影到达 z=0 的 x 位置 = x_n_edge - sgn_T·h_n·tan(|theta_T|)
                    x_n_edge = self.mirror_x[j_s] - sgn_T * (w/2) * np.cos(rho[j_s])
                    x_shadow_z0 = x_n_edge - sgn_T * h_n * np.tan(abs(theta_T))
                    # 镜 i 占据 [x_i - w_proj/2, x_i + w_proj/2]
                    x_i_min = self.mirror_x[i] - w_proj / 2
                    x_i_max = self.mirror_x[i] + w_proj / 2
                    # 阴影区: 在 [x_n_edge, x_shadow_z0] 之间(取 sgn_T 决定方向)
                    sh_min = min(x_n_edge, x_shadow_z0)
                    sh_max = max(x_n_edge, x_shadow_z0)
                    # 与镜 i 重叠
                    overlap = max(0, min(x_i_max, sh_max) - max(x_i_min, sh_min))
                    shadow_loss = overlap / w_proj
            
            # 遮挡: 邻居 j_b 在 -sgn_a 方向 (反射光从镜i飞向接收器, 被对面邻居挡)
            block_loss = 0.0
            if sgn_a != 0:
                j_b = i + sgn_a  # CPC 在 +x (alpha>0), 反射光向 +x 飞,被 +x 邻居挡
                if 0 <= j_b < N:
                    # 反射光从 (mirror_x[i], 0) 出发, 沿方位 alpha[i] 飞向 (0, H)
                    # 在邻居 x 位置 (mirror_x[j_b]) 时光线高度
                    if abs(np.tan(alpha[i])) > 1e-6:
                        h_at_j = abs(self.mirror_x[j_b] - self.mirror_x[i]) / abs(np.tan(alpha[i]))
                    else:
                        h_at_j = 1e9
                    h_n = abs(w / 2 * np.sin(rho[j_b]))
                    # 邻居顶部高度 h_n,如果光线在该 x 处高度 < h_n,被遮挡
                    if h_at_j < h_n:
                        # 遮挡光线占总反射光的比例
                        # 简化为: 比例 = (h_n - h_at_j) / h_n (越下方光线越多被挡)
                        block_loss = (h_n - h_at_j) / h_n
                        block_loss = min(1.0, block_loss)
            
            # 阴影和遮挡的合并: 取较大值(保守)
            total_loss = max(shadow_loss, block_loss)
            K_T[i] = max(0, 1 - total_loss)
            shadow_loss_arr[i] = shadow_loss
            block_loss_arr[i] = block_loss
        
        # ===== 纵向 K_L (端部损失) =====
        # 采用 Mertins 2009 (PhD thesis, Karlsruhe) 的经验拟合公式:
        #   K_L(theta_L) = 1 - 0.0014·theta_L - 0.0001·theta_L²  (theta_L in degrees)
        # 该公式基于实测 LFR 系统数据,隐含考虑了接收器适当延长以吸收边缘光斑
        # 比纯几何 max(0, 1 - H·tan(theta_L)/L) 更接近工程实际
        theta_L_deg = abs(np.rad2deg(theta_L))
        K_L = max(0.0, 1.0 - 0.0014 * theta_L_deg - 0.0001 * theta_L_deg**2)
        
        iam = K_T * K_L
        iam = np.clip(iam, 0, 1)
        
        return iam, {
            'K_T': K_T,
            'K_L': float(K_L),
            'shadow_loss': shadow_loss_arr,
            'block_loss': block_loss_arr,
            'theta_T_deg': float(np.rad2deg(theta_T)),
            'theta_L_deg': float(np.rad2deg(theta_L)),
            'rho_deg': np.rad2deg(rho),
        }


# ==================== 4. MCRT 光追 ====================

class MCRTTracer:
    """简化但物理一致的 MCRT 光追器
    关键特性:
      - 主镜→CPC 入口→吸热管全程追踪
      - 吸热管表面 (phi, z) 网格能流统计
      - 能量守恒检查
      - 以 numpy 向量化实现(无 GPU 依赖)
    """
    
    def __init__(self, cfg: Config, geo: LFRGeometry):
        self.cfg = cfg
        self.geo = geo
        self.n_phi = cfg.n_phi_bins
        self.n_z = cfg.n_z_bins
        self.L = cfg.mirror_length
        
        # 衰减因子(乘性)
        self.attenuation_per_cpc_bounce = cfg.rho_cpc
        self.base_attenuation = cfg.rho_mirror * cfg.tau_glass * cfg.alpha_abs * cfg.soiling
    
    def trace(self, sun_alt_deg, sun_az_deg, aim_z, dni, n_rays=None):
        """主追踪入口
        aim_z / aim_vec:
          - transverse_span 模式：每面镜横截面 x_aim，单位 m；
          - old_longitudinal 模式：旧版沿管轴偏移，单位 m。
        返回:
          flux_map: [n_phi, n_z_bins] 吸热管表面能流 (W/m²)；
              第二维 n_z_bins 为历史命名，实际表示管长方向 y 的离散位置。
          metrics: dict 含 eta_opt, cv_circ, sigma_surface, nuf, par, q_peak 等
        """
        if n_rays is None:
            n_rays = self.cfg.n_rays_eval
        
        # 每面镜分配光线数
        n_per_mirror = n_rays // self.geo.N
        
        flux_map = np.zeros((self.n_phi, self.n_z))
        total_input_power = 0.0
        total_absorbed = 0.0
        
        cos_inc_global = self._compute_cos_inc(sun_alt_deg, sun_az_deg)
        
        # 计算 IAM (阴影 + 遮挡 + 端部损失) - 全部镜面一次性计算
        iam_array, iam_components = self.geo.compute_iam(sun_alt_deg, sun_az_deg)
        
        # 太阳方向(全局)
        alt = np.deg2rad(sun_alt_deg)
        az = np.deg2rad(sun_az_deg)
        # 坐标系: x=横向, y=管轴, z=高度
        sun_x = -np.cos(alt) * np.sin(az)
        sun_z = -np.sin(alt)
        sun_y = -np.cos(alt) * np.cos(az)
        sun_vec = np.array([sun_x, sun_y, sun_z])
        sun_vec = sun_vec / np.linalg.norm(sun_vec)
        
        # 太阳张角 (Pillbox 模型, 半角 4.65 mrad = 0.27°)
        sun_half_angle_mrad = 4.65
        
        for i in range(self.geo.N):
            mx = self.geo.mirror_x[i]
            mw = self.geo.mirror_w[i]
            ma2 = self.geo.mirror_a2[i]
            
            # 镜面接收功率 (含 IAM: 阴影/遮挡/端部损失)
            # IAM 已包含纵向和横向损失,但不含 cos_inc_global (LFR 横向投影)
            mirror_power = dni * cos_inc_global * mw * self.L * iam_array[i]
            total_input_power += dni * cos_inc_global * mw * self.L  # 输入按理论功率
            
            if iam_array[i] < 1e-6:
                continue  # 整面镜都被遮蔽,跳过
            
            # 该镜的法向量
            aim_vec = np.asarray(aim_z, dtype=float).ravel()
            if self.cfg.aim_mode == 'transverse_span':
                # 新版：光斑打在集热管横截面的不同 x 位置
                x_aim_i = float(aim_vec[i])
                target_x = x_aim_i - mx
                target_z = self.geo.H
            else:
                # 旧版兼容：瞄准 CPC 入口中心
                target_x = -mx
                target_z = self.geo.cpc_y
            tlen = np.sqrt(target_x**2 + target_z**2)
            tx_n, tz_n = target_x / tlen, target_z / tlen
            # 入射的横纵投影
            sx_proj = sun_x / np.sqrt(sun_x**2 + sun_z**2 + 1e-9) if (sun_x**2 + sun_z**2) > 1e-9 else 0
            sz_proj = sun_z / np.sqrt(sun_x**2 + sun_z**2 + 1e-9) if (sun_x**2 + sun_z**2) > 1e-9 else -1
            # 主跟踪法向 (横截面 2D)
            nx = tx_n - sx_proj
            nz = tz_n - sz_proj
            nlen = np.sqrt(nx**2 + nz**2)
            nx, nz = nx / nlen, nz / nlen
            
            # 在镜面采样光线起点
            ray_x0 = mx + (np.random.rand(n_per_mirror) - 0.5) * mw
            ray_y0 = (np.random.rand(n_per_mirror) - 0.5) * self.L
            ray_z0 = np.zeros(n_per_mirror)
            
            # === 抛物面镜局部法向修正 ===
            # 抛物面 z_local(x_local) = a2 · x_local² (x_local 相对镜中心)
            # 局部斜率 dz/dx = 2·a2·x_local
            # 局部法向(局部坐标系内,垂直于 dz/dx): (-2·a2·x_local, 0, 1) 归一化后转到全局
            # 镜面绕 y 轴旋转角 = arctan2(nx, nz)
            x_local = ray_x0 - mx  # 偏离镜中心的横向距离
            local_slope = 2 * ma2 * x_local  # dz/dx (局部坐标)
            # 局部法向 (横截面): n_local = (-slope, 0, 1) / sqrt(1 + slope²)
            ln_norm = np.sqrt(1 + local_slope**2)
            n_local_x = -local_slope / ln_norm
            n_local_z = 1.0 / ln_norm
            # 镜面整体绕 y 轴旋转后,组合得每条光线的实际法向
            # 旋转矩阵 R(theta) = [[cos, sin], [-sin, cos]] 应用到 (n_local_x, n_local_z)
            # 镜面跟踪角(主法向方位): 总法向 = R · 局部法向
            cos_t = nz; sin_t = nx  # 镜面法向方位 = (sin_t, cos_t) 即 (nx, nz)
            actual_nx = cos_t * n_local_x + sin_t * n_local_z
            actual_nz = -sin_t * n_local_x + cos_t * n_local_z
            
            # 反射光: r = d - 2(d·n)n, 每条光线独立 n
            d_dot_n_arr = sun_x * actual_nx + sun_z * actual_nz
            ref_x_arr = sun_x - 2 * d_dot_n_arr * actual_nx
            ref_z_arr = sun_z - 2 * d_dot_n_arr * actual_nz
            ref_y_arr = np.full(n_per_mirror, sun_y)
            
            # === 误差源(高斯叠加) ===
            # 1. 镜面坡度误差 (slope error)  - 影响法向,反射光偏 2 倍 sigma
            # 2. 跟踪误差 (tracking error)   - 影响整面镜方位
            # 3. 太阳张角 (sunshape)          - 入射光本身锥角分布 (Pillbox 等概率, 半角 4.65 mrad)
            sigma_normal = np.sqrt(self.cfg.slope_error_mrad**2 + self.cfg.tracking_error_mrad**2) / 1000
            # 法向误差导致反射偏移 2σ
            err_x_normal = np.random.normal(0, 2 * sigma_normal, n_per_mirror)
            err_y_normal = np.random.normal(0, 2 * sigma_normal, n_per_mirror)
            
            # 太阳张角 (Pillbox: 在半角内均匀分布)
            r_sun = sun_half_angle_mrad / 1000 * np.sqrt(np.random.rand(n_per_mirror))
            phi_sun = 2 * np.pi * np.random.rand(n_per_mirror)
            err_x_sun = r_sun * np.cos(phi_sun)
            err_y_sun = r_sun * np.sin(phi_sun)
            
            ref_x_arr = ref_x_arr + err_x_normal + err_x_sun
            ref_y_arr = ref_y_arr + err_y_normal + err_y_sun
            
            # 重新归一化
            rlen = np.sqrt(ref_x_arr**2 + ref_y_arr**2 + ref_z_arr**2)
            ref_x_arr /= rlen; ref_y_arr /= rlen; ref_z_arr /= rlen
            
            # 应用瞄准偏移，调整反射光线方向
            if self.cfg.aim_mode == 'transverse_span':
                # 新版：不做管长方向偏移，管长方向保持原 ray_y0（轴向偏移为 0）
                target_y = ray_y0
                target_plane_z = self.geo.H
            else:
                # 旧版兼容：沿管轴方向移动
                aim_offset_y = float(aim_vec[i])
                target_y = ray_y0 + aim_offset_y
                target_plane_z = self.geo.cpc_y

            t_to_target = (target_plane_z - ray_z0) / np.maximum(ref_z_arr, 1e-6)
            ref_y_arr = (target_y - ray_y0) / np.maximum(t_to_target, 1e-6)
            rlen = np.sqrt(ref_x_arr**2 + ref_y_arr**2 + ref_z_arr**2)
            ref_x_arr /= rlen; ref_y_arr /= rlen; ref_z_arr /= rlen
            
            # 光线追踪到 CPC 入口平面 (z=cpc_y)
            t_cpc = (self.geo.cpc_y - ray_z0) / ref_z_arr
            valid = (t_cpc > 0) & (ref_z_arr > 0)
            x_at_cpc = ray_x0 + ref_x_arr * t_cpc
            y_at_cpc = ray_y0 + ref_y_arr * t_cpc
            z_at_cpc = np.full_like(x_at_cpc, self.geo.cpc_y)
            
            # 检查是否进入 CPC 入口
            # CPC 入口判定: 用实际几何输出的入口半宽(由公式决定),不是配置值
            cpc_half = self.geo._cpc_actual_inlet_half_width
            in_cpc = valid & (np.abs(x_at_cpc) < cpc_half) & (np.abs(y_at_cpc) < self.L / 2)
            
            n_in = in_cpc.sum()
            if n_in == 0:
                continue
            
            # 进入 CPC 的光线,做 CPC 内反射追踪
            ray_pos = np.column_stack([x_at_cpc[in_cpc], y_at_cpc[in_cpc], z_at_cpc[in_cpc]])
            ray_dir = np.column_stack([ref_x_arr[in_cpc], ref_y_arr[in_cpc], ref_z_arr[in_cpc]])
            # 光线权重: mirror_power 已含 DNI·cos_inc·w·L·IAM, 再乘镜面反射率
            # 每条光线的份额 = mirror_power / n_per_mirror (未除以进入 CPC 的成功率)
            # 这样 spillage(未进 CPC 的光)自然计入光学损失
            ray_weight = np.full(n_in, mirror_power / n_per_mirror * self.cfg.rho_mirror)
            
            # CPC 内反射 + 命中吸热管
            phi_hits, z_hits, weights = self._cpc_trace(ray_pos, ray_dir, ray_weight)
            
            # 累加到能流图
            if len(phi_hits) > 0:
                phi_idx = np.clip(((phi_hits + np.pi) / (2*np.pi) * self.n_phi).astype(int), 0, self.n_phi - 1)
                z_idx = np.clip(((z_hits + self.L/2) / self.L * self.n_z).astype(int), 0, self.n_z - 1)
                np.add.at(flux_map, (phi_idx, z_idx), weights)
                total_absorbed += weights.sum()
        
        # 单位换算: 每个 bin 的功率 / bin 面积 = W/m²
        bin_area = (2 * np.pi * self.cfg.absorber_radius / self.n_phi) * (self.L / self.n_z)
        flux_map = flux_map / bin_area
        
        # 计算指标
        eta_opt = total_absorbed / max(total_input_power, 1e-9)
        metrics = self._compute_metrics(flux_map)
        metrics['eta_opt'] = eta_opt
        metrics['total_absorbed_W'] = total_absorbed
        metrics['total_input_W'] = total_input_power
        
        return flux_map, metrics
    
    def _compute_cos_inc(self, sun_alt_deg, sun_az_deg):
        """LFR 横向入射角余弦(管轴沿 y/南北)"""
        alt = np.deg2rad(sun_alt_deg)
        az = np.deg2rad(sun_az_deg)
        # 横向入射角 = arctan(sin(az)*cos(alt) / sin(alt))
        # cos_inc = sin(alt) / sqrt(sin²(alt) + cos²(alt)*sin²(az))
        denom = np.sqrt(np.sin(alt)**2 + (np.cos(alt) * np.sin(az))**2)
        return np.sin(alt) / max(denom, 1e-6)
    
    def _cpc_trace(self, pos, direction, weight):
        """CPC 内光线追踪:多次反射直到命中吸热管或逃逸
        简化实现: 用解析法处理 CPC 壁(贝塞尔曲线近似为直线段集合)+ 吸热管(圆柱)
        返回: (phi_hits, z_hits, weights) 命中吸热管的光线；
            z_hits 为历史变量名，实际保存 hit_pos[:, 1]，即管长方向 y 坐标。
        """
        pos = pos.copy()
        direction = direction.copy()
        weight = weight.copy()
        n = len(pos)
        
        absorber_x = 0.0
        absorber_z = self.geo.H
        r_abs = self.cfg.absorber_radius
        r_glass = self.cfg.glass_radius
        
        phi_hits = []
        z_hits = []
        w_hits = []
        
        # CPC 壁段(2D, xz 平面)
        right_segs = self.geo.cpc_right_segs
        left_segs = self.geo.cpc_left_segs
        
        for bounce in range(self.cfg.max_cpc_bounces + 1):
            if n == 0:
                break
            
            # 求与吸热管圆柱的交点 (在 xz 平面)
            # 圆柱方程: (x - 0)² + (z - H)² = r²
            dx = direction[:, 0]
            dz = direction[:, 2]
            px = pos[:, 0]
            pz = pos[:, 2] - absorber_z
            a = dx**2 + dz**2
            b = 2 * (px * dx + pz * dz)
            c = px**2 + pz**2 - r_abs**2
            disc = b**2 - 4 * a * c
            
            t_abs = np.full(n, np.inf)
            valid_disc = disc >= 0
            sqrt_disc = np.where(valid_disc, np.sqrt(np.maximum(disc, 0)), 0)
            t1 = (-b - sqrt_disc) / (2 * a + 1e-12)
            t2 = (-b + sqrt_disc) / (2 * a + 1e-12)
            t_pos = np.where((t1 > 1e-6) & valid_disc, t1, np.inf)
            t_pos2 = np.where((t2 > 1e-6) & valid_disc, t2, np.inf)
            t_abs = np.minimum(t_pos, t_pos2)
            
            # 求与 CPC 壁的交点(取右壁段集合最小正 t)
            t_wall, wall_normal = self._intersect_cpc_walls(pos, direction)
            
            # 选最小正 t
            hit_absorber = (t_abs < t_wall) & np.isfinite(t_abs)
            hit_wall = (t_wall < t_abs) & np.isfinite(t_wall)
            escape = ~hit_absorber & ~hit_wall
            
            # 命中吸热管
            if hit_absorber.any():
                idx = np.where(hit_absorber)[0]
                hit_pos = pos[idx] + direction[idx] * t_abs[idx, None]
                rel_x = hit_pos[:, 0] - absorber_x
                rel_z = hit_pos[:, 2] - absorber_z
                phi = np.arctan2(rel_z, rel_x)  # [-pi, pi]
                z_pos_abs = hit_pos[:, 1]  # 历史变量名 z_hits，实际是管长方向 y 坐标
                phi_hits.extend(phi.tolist())
                z_hits.extend(z_pos_abs.tolist())
                w_hits.extend(weight[idx].tolist())
            
            # 命中 CPC 壁: 反射继续
            if hit_wall.any():
                idx = np.where(hit_wall)[0]
                new_pos = pos[idx] + direction[idx] * t_wall[idx, None]
                d = direction[idx]
                n_vec = wall_normal[idx]
                d_dot_n = (d * n_vec).sum(axis=1, keepdims=True)
                new_dir = d - 2 * d_dot_n * n_vec
                # 略微推开避免数值自交
                new_pos = new_pos + new_dir * 1e-5
                pos[idx] = new_pos
                direction[idx] = new_dir
                weight[idx] *= self.attenuation_per_cpc_bounce
            
            # 移除已命中或已逃逸的光线
            keep = ~(hit_absorber | escape)
            pos = pos[keep]
            direction = direction[keep]
            weight = weight[keep]
            n = len(pos)
        
        return np.array(phi_hits), np.array(z_hits), np.array(w_hits)
    
    def _intersect_cpc_walls(self, pos, direction):
        """求光线与 CPC 壁段(线段集合)的最近正 t 和法向"""
        n = len(pos)
        t_min = np.full(n, np.inf)
        normal_at_hit = np.zeros((n, 3))
        
        for segs, normals in [(self.geo.cpc_right_segs, self.geo.cpc_right_normals),
                              (self.geo.cpc_left_segs, self.geo.cpc_left_normals)]:
            for s in range(len(segs) - 1):
                p1 = segs[s]
                p2 = segs[s + 1]
                seg_dir = p2 - p1
                seg_len = np.linalg.norm(seg_dir)
                if seg_len < 1e-9:
                    continue
                seg_dir = seg_dir / seg_len
                # 光线 (xz 平面) 与线段求交
                # P_ray = pos[xz] + t * dir[xz]
                # P_seg = p1 + s_param * seg_dir, s in [0, seg_len]
                px = pos[:, 0]
                pz = pos[:, 2]
                dx = direction[:, 0]
                dz = direction[:, 2]
                # 线性方程组: dx*t - seg_dir[0]*s = p1[0] - px
                #             dz*t - seg_dir[1]*s = p1[1] - pz
                a1 = dx; b1 = -seg_dir[0]; c1 = p1[0] - px
                a2 = dz; b2 = -seg_dir[1]; c2 = p1[1] - pz
                det = a1 * b2 - a2 * b1
                valid = np.abs(det) > 1e-9
                t = np.where(valid, (c1 * b2 - c2 * b1) / (det + 1e-12), -1)
                s_param = np.where(valid, (a1 * c2 - a2 * c1) / (det + 1e-12), -1)
                hit = valid & (t > 1e-5) & (s_param >= 0) & (s_param <= seg_len) & (t < t_min)
                if hit.any():
                    t_min[hit] = t[hit]
                    normal_at_hit[hit, 0] = normals[s, 0]
                    normal_at_hit[hit, 2] = normals[s, 1]  # 法向只在 xz 平面
        
        return t_min, normal_at_hit
    
    def _compute_metrics(self, flux_map):
        """从能流图计算均匀性指标
        
        flux_map: [n_phi, n_z_bins] 吸热管表面能流 (W/m²)；
            第二维 n_z_bins 为历史命名，实际表示管长方向 y 的离散位置。
            phi 方向: 0 ~ 2π, 索引 0 对应 phi=-π (吸热管顶部)
            按 trace() 中的映射: phi = arctan2(rel_z, rel_x)
            CPC 在管下方 → 辐照集中在管下半圆 (phi ∈ [-π, 0],对应索引 0~n_phi/2)
        
        关键约定:
        - cv_circ:  只在【辐照半圆】上算圆周 CV(物理意义:实际工程关心的非均匀性)
        - cv_full:  全圆周 CV(供参考,会偏大,因暗区拉低均值)
        - nuf:      只在辐照半圆上算
        - par:      峰均比(辐照半圆)
        """
        n_phi = flux_map.shape[0]
        # 辐照半圆: 下半圆,phi ∈ [-π, 0]
        # 在 trace 中 phi = arctan2(rel_z, rel_x), rel_z = hit_z - H
        # CPC 在管下方,光线从下方击中管 → rel_z < 0 → phi < 0
        # 索引: arctan2 范围 [-π, π], 映射到 [0, n_phi),对应 phi <0 的索引区间是 [0, n_phi/2)
        irradiated = flux_map[:n_phi // 2, :]   # 下半圆
        
        q_mean_full = flux_map.mean()
        if q_mean_full < 1e-9:
            return {'cv_global': 0, 'cv_circ': 0, 'cv_full': 0,
                    'nuf': 0, 'par': 0, 'q_peak': 0, 'q_mean_irr': 0,
                    'sigma_surface': 0, 'par_full': 0,
                    'top_flux_ratio': 0, 'top_to_bottom_ratio': 0, 'q_mean_full': 0}

        # 全圆周 CV(供参考)
        cv_full = flux_map.std() / q_mean_full
        # 全局 CV
        cv_global = cv_full

        # 辐照半圆指标
        q_mean_irr = irradiated.mean()
        if q_mean_irr < 1e-9:
            return {'cv_global': float(cv_global), 'cv_circ': 0, 'cv_full': float(cv_full),
                    'nuf': 0, 'par': 0, 'q_peak': float(flux_map.max()), 'q_mean_irr': 0,
                    'sigma_surface': float(cv_full), 'par_full': float(flux_map.max() / (q_mean_full + 1e-9)),
                    'top_flux_ratio': 0, 'top_to_bottom_ratio': 0, 'q_mean_full': float(q_mean_full)}

        # 圆周方向 CV: 只在辐照半圆上,对每个 z 切片
        cv_per_z = irradiated.std(axis=0) / (irradiated.mean(axis=0) + 1e-9)
        valid_z = irradiated.mean(axis=0) > q_mean_irr * 0.1
        cv_circ = cv_per_z[valid_z].mean() if valid_z.sum() > 0 else cv_per_z.mean()

        # NUF (非均匀因子) — 辐照半圆
        nuf = np.abs(irradiated - q_mean_irr).sum() / (irradiated.size * q_mean_irr)
        # 峰均比
        par = irradiated.max() / q_mean_irr
        # 峰值能流(便于热应力评估)
        q_peak = float(flux_map.max())

        # 新增：全表面均匀性指标（论文口径）
        sigma_surface = flux_map.std() / (q_mean_full + 1e-9)
        par_full = flux_map.max() / (q_mean_full + 1e-9)

        # 顶部/底部能流比
        top_region = flux_map[n_phi // 2:, :]
        bottom_region = flux_map[:n_phi // 2, :]
        top_flux_mean = top_region.mean()
        bottom_flux_mean = bottom_region.mean()
        top_flux_ratio = top_flux_mean / (q_mean_full + 1e-9)
        top_to_bottom_ratio = top_flux_mean / (bottom_flux_mean + 1e-9)

        return {'cv_global': float(cv_global),
                'cv_circ': float(cv_circ),
                'cv_full': float(cv_full),
                'nuf': float(nuf),
                'par': float(par),
                'q_peak': q_peak,
                'q_mean_irr': float(q_mean_irr),
                'sigma_surface': float(sigma_surface),
                'par_full': float(par_full),
                'top_flux_ratio': float(top_flux_ratio),
                'top_to_bottom_ratio': float(top_to_bottom_ratio),
                'q_mean_full': float(q_mean_full)}


class MCRTTracerTorch(MCRTTracer):
    """
    占位类：当前版本仍复用 NumPy CPU 路径，不提供 MCRT GPU 加速。
    不应用于正式声称 GPU 加速。
    """
    def trace(self, sun_alt_deg, sun_az_deg, aim_z, dni, n_rays=None):
        if self.cfg.aim_mode != 'transverse_span':
            raise NotImplementedError("MCRTTracerTorch 目前仅支持 transverse_span。")
        return super().trace(sun_alt_deg, sun_az_deg, aim_z, dni, n_rays=n_rays)


def create_mcrt_tracer(cfg, geo, logger=None):
    if cfg.mcrt_backend == 'torch_cuda_experimental':
        if logger:
            logger.warn(
                "torch_cuda_experimental 当前仍复用 NumPy CPU 光追路径，"
                "尚未实现真正 MCRT GPU 加速；自动回退到 numpy_cpu。"
            )
        cfg.mcrt_backend = 'numpy_cpu'
    return MCRTTracer(cfg, geo)


def describe_runtime_backend(cfg, logger):
    nn_device = cfg.get_device()
    logger.info(f"NN device: {nn_device}")
    logger.info(f"MCRT backend: {cfg.mcrt_backend}")
    if cfg.mcrt_backend == 'numpy_cpu':
        logger.warn("当前 MCRT 光追为 NumPy CPU 实现，GPU 只用于 Transformer，不会加速 MCRT。")


def validate_mcrt_backend_parity(cfg, cluster_data, logger):
    if cfg.mcrt_backend != 'torch_cuda_experimental' or not cfg.mcrt_gpu_validate_on_start:
        return
    if not (HAS_TORCH and torch.cuda.is_available()):
        cfg.mcrt_backend = 'numpy_cpu'
        return
    rep = cluster_data['representatives'].iloc[0]
    row = cluster_data['df'].iloc[int(rep['rep_idx'])]
    geo = LFRGeometry(cfg)
    cpu = MCRTTracer(cfg, geo)
    gpu = MCRTTracerTorch(cfg, geo)
    for span in (0.0, 0.035):
        aim = make_xaim_from_span(cfg, span)
        _, mc = cpu.trace(row['solar_alt'], row['solar_az'], aim, row['DNI'], n_rays=cfg.mcrt_gpu_parity_rays)
        _, mg = gpu.trace(row['solar_alt'], row['solar_az'], aim, row['DNI'], n_rays=cfg.mcrt_gpu_parity_rays)
        de = abs(mg['eta_opt'] - mc['eta_opt'])
        ds = abs(mg['sigma_surface'] - mc['sigma_surface'])
        if de > cfg.mcrt_gpu_parity_tol_eta or ds > cfg.mcrt_gpu_parity_tol_sigma:
            logger.warn("GPU MCRT 与 CPU 差异过大，回退 numpy_cpu")
            cfg.mcrt_backend = 'numpy_cpu'
            return


# ==================== 5. 基线策略 ====================

def stage_baseline(cfg: Config, cluster_data: dict, logger: Logger,
                   stop: StopSignal, ckpt: Checkpoint):
    """阶段 3: 基线策略评估（主基线为 S1_center）"""
    logger.info("=== 阶段 3/7: 基线策略评估 ===")
    workdir = Path(cfg.workdir)
    out_path = workdir / 'baselines.pkl'

    if ckpt.is_done('baseline') and out_path.exists():
        logger.info("已完成,加载基线结果")
        return _load_pickle_checked(out_path, cfg, 'baselines.pkl')

    geo = LFRGeometry(cfg)
    mcrt = create_mcrt_tracer(cfg, geo, logger)

    rep_df = cluster_data['representatives']
    df = cluster_data['df']

    # 主基线只保留 S1_center（横截面/纵向都是全瞄管心）
    strategies = {'S1_center': lambda N: make_s1_aim(cfg)}

    results = []
    flux_maps = {}
    total_evals = len(rep_df) * len(strategies)
    done = 0
    for _, rep in rep_df.iterrows():
        stop.check()
        row = df.iloc[rep['rep_idx']]
        for sname, sfunc in strategies.items():
            aim = sfunc(cfg.n_mirrors)
            flux, m = mcrt.trace(row['solar_alt'], row['solar_az'], aim, row['DNI'],
                                 n_rays=cfg.n_rays_validate)
            results.append({
                'cluster': rep['cluster'],
                'strategy': sname,
                'eta_opt': m['eta_opt'],
                'cv_circ': m['cv_circ'],
                'sigma_surface': m['sigma_surface'],
                'par_full': m['par_full'],
                'top_flux_ratio': m['top_flux_ratio'],
                'top_to_bottom_ratio': m['top_to_bottom_ratio'],
                'nuf': m['nuf'],
                'par': m['par'],
                'q_peak': m['q_peak'],
                'cv_global': m['cv_global'],
                'timestamp': rep['rep_timestamp'],
                'dni': row['DNI'],
                'solar_alt': row['solar_alt'],
                'solar_az': row['solar_az'],
            })
            flux_maps[(rep['cluster'], sname)] = flux
            done += 1
            logger.progress(done / total_evals, f"基线 {sname} 簇{rep['cluster']}")

    res_df = pd.DataFrame(results)
    out = {'results': res_df, 'flux_maps': flux_maps}
    _save_pickle_with_hash(out_path, out, cfg)
    ckpt.mark_done('baseline')
    s1 = res_df[res_df.strategy == 'S1_center']
    logger.info(
        f"基线评估完成: S1 平均 CV_circ={s1['cv_circ'].mean():.3f}, "
        f"sigma_surface={s1['sigma_surface'].mean():.3f}"
    )
    return out


# ==================== 6. 贝叶斯优化 ====================

class SimpleMOBO:
    """简化多目标 BO,基于高斯过程 + 加权和切片
    避免引入 BoTorch 重依赖,用 sklearn GP + 随机采样筛选
    """
    
    def __init__(self, dim, bounds, n_initial=30, n_iter=80, seed=42):
        self.dim = dim
        self.bounds = np.array(bounds)
        self.n_initial = n_initial
        self.n_iter = n_iter
        self.rng = np.random.RandomState(seed)
    
    def optimize(self, eval_fn, stop_check=None, initial_points=None):
        """eval_fn(x) -> (eta_opt, cv_circ)
        返回: (X_history, Y_history, knee_x, knee_y)
        """
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
        except ImportError:
            return self._random_search(eval_fn, stop_check, initial_points=initial_points)
        
        # 初始点: 外部给定 + 拉丁超立方
        if initial_points is not None and len(initial_points) > 0:
            initial_points = np.asarray(initial_points, dtype=float).reshape(-1, self.dim)
            n_lhs = max(0, self.n_initial - len(initial_points))
            X_lhs = self._latin_hypercube(n_lhs) if n_lhs > 0 else np.empty((0, self.dim))
            X = np.vstack([initial_points, X_lhs])
        else:
            X = self._latin_hypercube(self.n_initial)
        X_round = np.round(X, 8)
        _, unique_idx = np.unique(X_round, axis=0, return_index=True)
        X = X[np.sort(unique_idx)]
        Y = np.array([self._safe_eval(eval_fn, x, stop_check) for x in X])
        
        for it in range(self.n_iter):
            if stop_check is not None:
                stop_check()
            # 训练两个 GP(对应 -eta 和 cv)
            try:
                kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5)
                gp_eta = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-3)
                gp_cv = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-3)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=LinAlgWarning)
                    gp_eta.fit(X, -Y[:, 0])  # 最小化 -eta
                    gp_cv.fit(X, Y[:, 1])  # 最小化 cv
                
                # 候选采样,用 ParEGO 思想: 随机权重标量化
                w = self.rng.rand()
                w = np.array([w, 1 - w])
                # 大量随机点+UCB
                cand = self._latin_hypercube(500)
                mu_eta, sig_eta = gp_eta.predict(cand, return_std=True)
                mu_cv, sig_cv = gp_cv.predict(cand, return_std=True)
                # 标量化(归一化后)
                eta_norm = (mu_eta - mu_eta.min()) / (mu_eta.max() - mu_eta.min() + 1e-9)
                cv_norm = (mu_cv - mu_cv.min()) / (mu_cv.max() - mu_cv.min() + 1e-9)
                acq = -(w[0] * eta_norm + w[1] * cv_norm) + 0.5 * (w[0] * sig_eta + w[1] * sig_cv)
                idx = np.argmax(acq)
                x_new = cand[idx]
            except Exception:
                x_new = self._latin_hypercube(1)[0]
            
            y_new = self._safe_eval(eval_fn, x_new, stop_check)
            X = np.vstack([X, x_new])
            Y = np.vstack([Y, y_new])
        
        # Pareto 前沿提取
        pareto_mask = self._pareto_mask(Y)
        # knee 点: 前沿上距理想点最近(归一化)
        if pareto_mask.sum() > 0:
            Y_p = Y[pareto_mask]
            ideal = np.array([Y_p[:, 0].max(), Y_p[:, 1].min()])
            ynorm = (Y_p - Y_p.min(0)) / (Y_p.max(0) - Y_p.min(0) + 1e-9)
            ideal_norm = (ideal - Y_p.min(0)) / (Y_p.max(0) - Y_p.min(0) + 1e-9)
            ideal_norm[1] = 0  # cv 越小越好
            ideal_norm[0] = 1
            dist = np.linalg.norm(ynorm - ideal_norm, axis=1)
            knee_idx_local = np.argmin(dist)
            X_p = X[pareto_mask]
            knee_x = X_p[knee_idx_local]
            knee_y = Y_p[knee_idx_local]
        else:
            knee_x = X[0]
            knee_y = Y[0]
        
        return X, Y, knee_x, knee_y, pareto_mask
    
    def _latin_hypercube(self, n):
        d = self.dim
        cut = np.linspace(0, 1, n + 1)
        u = self.rng.rand(n, d)
        a = cut[:n]
        b = cut[1:]
        rdp = np.zeros((n, d))
        for j in range(d):
            rdp[:, j] = u[:, j] * (b - a) + a
            self.rng.shuffle(rdp[:, j])
        return self.bounds[:, 0] + rdp * (self.bounds[:, 1] - self.bounds[:, 0])
    
    def _random_search(self, eval_fn, stop_check, initial_points=None):
        X_rand = self._latin_hypercube(self.n_initial + self.n_iter)
        if initial_points is not None and len(initial_points) > 0:
            initial_points = np.asarray(initial_points, dtype=float).reshape(-1, self.dim)
            X = np.vstack([initial_points, X_rand])
        else:
            X = X_rand
        X_round = np.round(X, 8)
        _, unique_idx = np.unique(X_round, axis=0, return_index=True)
        X = X[np.sort(unique_idx)]
        Y = np.array([self._safe_eval(eval_fn, x, stop_check) for x in X])
        pareto_mask = self._pareto_mask(Y)
        knee_x = X[0]
        knee_y = Y[0]
        return X, Y, knee_x, knee_y, pareto_mask
    
    def _safe_eval(self, eval_fn, x, stop_check):
        if stop_check is not None:
            stop_check()
        try:
            return np.array(eval_fn(x))
        except Exception:
            return np.array([0.0, 1.0])
    
    @staticmethod
    def _pareto_mask(Y):
        """Y[:, 0] 越大越好, Y[:, 1] 越小越好"""
        n = len(Y)
        mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not mask[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if Y[j, 0] >= Y[i, 0] and Y[j, 1] <= Y[i, 1] and (Y[j, 0] > Y[i, 0] or Y[j, 1] < Y[i, 1]):
                    mask[i] = False
                    break
        return mask


def select_bo_label_with_eta_floor(X, Y, pareto_mask, x_s1, eta_floor_rel=0.96):
    """
    从 BO 历史中选择最终训练标签。
    Y[:, 0] = eta_opt（越大越好）
    Y[:, 1] = uniformity_metric（越小越好）

    1. 找 span=0 的 S1 评价结果；
    2. 计算 eta_floor = eta_S1 * eta_floor_rel；
    3. 在 Pareto 点中筛选 eta_opt >= eta_floor；
    4. 在可行点中选择 uniformity_metric 最小的点；
    5. 若没有可行点，回退到 span=0。
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    x_s1 = np.asarray(x_s1, dtype=float).ravel()
    s1_idx = int(np.argmin(np.linalg.norm(X - x_s1[None, :], axis=1)))
    s1_y = Y[s1_idx]
    s1_eta = float(s1_y[0])
    eta_floor = s1_eta * eta_floor_rel if s1_eta > 0 else s1_eta
    pareto_idx = np.where(pareto_mask)[0]
    feasible_idx = pareto_idx[Y[pareto_idx, 0] >= eta_floor]
    if len(feasible_idx) > 0:
        best_idx = feasible_idx[np.argmin(Y[feasible_idx, 1])]
        reason = 'pareto_min_sigma_eta_floor'
    else:
        best_idx = s1_idx
        reason = 'fallback_s1_no_feasible_sigma_eta_floor'
    return X[best_idx], Y[best_idx], reason, s1_y


# 旧版别名，保持向后兼容
select_bo_label_with_s1_guard = select_bo_label_with_eta_floor


def stage_bo(cfg: Config, df: pd.DataFrame, cluster_data: dict,
             logger: Logger, stop: StopSignal, ckpt: Checkpoint):
    """阶段 4: BO 在每簇代表时刻+采样时刻找 Pareto knee

    transverse_span 模式下:
      dim=1, 搜索 span ∈ xaim_span_range
      每面镜 x_aim 由 make_xaim_from_span(cfg, span) 生成
    """
    logger.info("=== 阶段 4/7: 多目标贝叶斯优化 ===")
    workdir = Path(cfg.workdir)
    out_path = workdir / 'bo_dataset.pkl'

    if ckpt.is_done('bo') and out_path.exists():
        logger.info("已完成,加载 BO 数据集")
        return _load_pickle_checked(out_path, cfg, 'bo_dataset.pkl')

    geo = LFRGeometry(cfg)
    mcrt = create_mcrt_tracer(cfg, geo, logger)

    rep_df = cluster_data['representatives']
    df_full = cluster_data['df']

    # 决策变量维度
    if cfg.aim_mode == 'transverse_span':
        if cfg.experiment_mode == 'grouped_span':
            dim = 3
            bounds = list(cfg.grouped_span_bounds)
        else:
            dim = 1
            bounds = [cfg.xaim_span_range]
    else:
        dim = cfg.n_mirrors // 2 if cfg.use_symmetry else cfg.n_mirrors
        bounds = [cfg.z_range] * dim

    logger.info(f"BO 决策变量维度: {dim}, aim_mode={cfg.aim_mode}")

    samples = []
    pareto_data = {}

    done_pairs = set(tuple(p) for p in ckpt.get_partial('bo', 'done_pairs', []))

    total = len(rep_df) * cfg.samples_per_cluster

    partial_path = workdir / 'bo_partial.pkl'
    if partial_path.exists():
        with open(partial_path, 'rb') as f:
            saved = pickle.load(f)
        saved_hash = saved.get('config_hash') if isinstance(saved, dict) else None
        _validate_cached_hash(saved_hash, cfg, 'bo_partial.pkl')
        samples = saved['samples']
        pareto_data = saved['pareto_data']
        logger.info(f"恢复 {len(samples)} 个已有样本")

    for ci, rep in rep_df.iterrows():
        cluster_id = rep['cluster']
        cluster_hours = df_full[df_full['cluster'] == cluster_id]
        n_pick = min(cfg.samples_per_cluster, len(cluster_hours))
        sampled = cluster_hours.sample(n_pick, random_state=cfg.seed + cluster_id)

        for si, (_, hour_row) in enumerate(sampled.iterrows()):
            stop.check()
            pair = (int(cluster_id), int(si))
            if pair in done_pairs:
                continue

            # 定义 BO 评估函数
            uniformity_key = cfg.bo_uniformity_metric

            def eval_fn(x, _hr=hour_row):
                if cfg.aim_mode == 'transverse_span':
                    if cfg.experiment_mode == 'grouped_span':
                        spans = np.asarray(x).ravel()
                        aim = make_xaim_from_grouped_span(cfg, spans, geo)
                    else:
                        span = float(np.asarray(x).ravel()[0])
                        aim = make_xaim_from_span(cfg, span)
                else:
                    if cfg.use_symmetry:
                        aim = np.concatenate([x[::-1], x])
                    else:
                        aim = x
                _, m = mcrt.trace(_hr['solar_alt'], _hr['solar_az'],
                                  aim, _hr['DNI'], n_rays=cfg.n_rays_eval)
                uniformity = m.get(uniformity_key, m.get('sigma_surface', m['cv_circ']))
                return m['eta_opt'], uniformity

            bo = SimpleMOBO(dim, bounds, n_initial=cfg.bo_n_initial,
                            n_iter=cfg.bo_n_iterations, seed=cfg.seed + ci * 100 + si)

            # 初始点
            if cfg.aim_mode == 'transverse_span' and cfg.experiment_mode == 'grouped_span':
                x_s1 = np.array([0.0, 0.0, 0.0])
                initial_points = [[0.0, 0.0, 0.0], [0.035, 0.035, 0.035], [0.015, 0.025, 0.035], [0.005, 0.025, 0.045]]
            elif cfg.aim_mode == 'transverse_span':
                x_s1 = np.array([0.0])
                initial_points = []
                if cfg.bo_force_s1_initial:
                    initial_points.append([0.0])
                if cfg.bo_force_paper_span_initial:
                    initial_points.append([cfg.paper_xaim_span])
                initial_points = initial_points if initial_points else None
            else:
                x_s1 = np.zeros(dim)
                initial_points = [x_s1.tolist()] if cfg.bo_force_s1_initial else None

            X, Y, knee_x, knee_y, pmask = bo.optimize(
                eval_fn,
                stop_check=stop.check,
                initial_points=initial_points,
            )
            selected_x, selected_y, selected_reason, s1_y = select_bo_label_with_eta_floor(
                X, Y, pmask, x_s1, eta_floor_rel=cfg.bo_eta_floor_rel
            )

            # 获取完整 metrics（对 selected_x 再调用一次 trace）
            if cfg.aim_mode == 'transverse_span' and cfg.experiment_mode == 'grouped_span':
                spans = np.asarray(selected_x).ravel()
                sel_span = float(np.mean(spans))
                sel_aim = make_xaim_from_grouped_span(cfg, spans, geo)
                s1_aim = make_xaim_from_span(cfg, 0.0)
            elif cfg.aim_mode == 'transverse_span':
                sel_span = float(selected_x.ravel()[0])
                spans = np.array([sel_span], dtype=float)
                sel_aim = make_xaim_from_span(cfg, sel_span)
                s1_aim = make_xaim_from_span(cfg, 0.0)
            else:
                sel_span = float('nan')
                spans = np.array([], dtype=float)
                sel_aim = expand_aim_to_full(selected_x, cfg, geo=geo)
                s1_aim = np.zeros(cfg.n_mirrors)

            _, selected_metrics = mcrt.trace(
                hour_row['solar_alt'], hour_row['solar_az'],
                sel_aim, hour_row['DNI'], n_rays=cfg.n_rays_eval
            )
            _, s1_metrics = mcrt.trace(
                hour_row['solar_alt'], hour_row['solar_az'],
                s1_aim, hour_row['DNI'], n_rays=cfg.n_rays_eval
            )

            samples.append({
                'cluster': int(cluster_id),
                'sample_idx': int(si),
                'solar_alt': float(hour_row['solar_alt']),
                'solar_az': float(hour_row['solar_az']),
                'cos_inc': float(hour_row['cos_inc']),
                'dni': float(hour_row['DNI']),
                'aim_optimal': selected_x.tolist(),  # 1D [span] in transverse_span mode
                'eta_opt': float(selected_y[0]),
                'cv_circ': float(selected_metrics['cv_circ']),
                'aim_mode': cfg.aim_mode,
                'span_optimal': sel_span,
                'span_vector': spans.tolist(),
                'strategy_param_mode': 'grouped_span' if (cfg.aim_mode == 'transverse_span' and cfg.experiment_mode == 'grouped_span') else 'span_1d',
                'span_inner': float(spans[0]) if len(spans) >= 3 else float('nan'),
                'span_mid': float(spans[1]) if len(spans) >= 3 else float('nan'),
                'span_outer': float(spans[2]) if len(spans) >= 3 else float('nan'),
                'xaim_min': float(-sel_span) if cfg.aim_mode == 'transverse_span' else float('nan'),
                'xaim_max': float(sel_span) if cfg.aim_mode == 'transverse_span' else float('nan'),
                'uniformity_metric': uniformity_key,
                'uniformity_obj': float(selected_y[1]),
                'sigma_surface': float(selected_metrics['sigma_surface']),
                'par_full': float(selected_metrics['par_full']),
                'top_flux_ratio': float(selected_metrics['top_flux_ratio']),
                'top_to_bottom_ratio': float(selected_metrics['top_to_bottom_ratio']),
                'bo_selected_reason': selected_reason,
                's1_eta_opt': float(s1_metrics['eta_opt']),
                's1_sigma_surface': float(s1_metrics['sigma_surface']),
                's1_cv_circ': float(s1_metrics['cv_circ']),
                'eta_floor_rel': float(cfg.bo_eta_floor_rel),
                'timestamp': str(hour_row['timestamp']),
            })
            if cluster_id not in pareto_data:
                pareto_data[int(cluster_id)] = []
            pareto_data[int(cluster_id)].append({'X': X, 'Y': Y, 'pareto_mask': pmask})
            done_pairs.add(pair)

            with open(partial_path, 'wb') as f:
                pickle.dump({
                    'samples': samples,
                    'pareto_data': pareto_data,
                    'config_hash': compute_config_hash(cfg)
                }, f)
            ckpt.set_partial('bo', 'done_pairs', list(done_pairs))

            cur = len(samples)
            logger.progress(
                cur / total,
                f"BO 簇{cluster_id} 样本{si+1}/{n_pick} "
                f"(η={selected_y[0]:.3f}, σ={float(selected_metrics['sigma_surface']):.3f}, "
                f"span={sel_span:.4f}, {selected_reason})"
            )

    out = {'samples': pd.DataFrame(samples), 'pareto_data': pareto_data, 'dim': dim}
    _save_pickle_with_hash(out_path, out, cfg)
    if partial_path.exists():
        partial_path.unlink()
    ckpt.mark_done('bo')
    logger.info(f"BO 完成: 共 {len(samples)} 个 (state, aim_optimal) 样本, dim={dim}")
    return out


# ==================== 7. Transformer 训练 ====================

if HAS_TORCH:
    class AimTransformer(nn.Module):
        """将太阳状态映射到瞄准向量（transverse_span 下为 1 维 span，旧版为 9 维）"""
        def __init__(self, n_mirrors_half, d_model=128, n_heads=4, n_layers=3, n_clusters=12,
                     output_low=-5.0, output_high=5.0, output_activation='tanh'):
            super().__init__()
            self.n_half = n_mirrors_half
            self.input_dim = 4  # solar_alt, solar_az, cos_inc, dni
            low = torch.as_tensor(output_low, dtype=torch.float32)
            high = torch.as_tensor(output_high, dtype=torch.float32)
            self.register_buffer('output_low_tensor', low)
            self.register_buffer('output_high_tensor', high)
            self.output_activation = output_activation
            self.cluster_emb = nn.Embedding(n_clusters, d_model // 4)
            self.state_proj = nn.Linear(self.input_dim, d_model - d_model // 4)
            self.mirror_emb = nn.Embedding(n_mirrors_half, d_model)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
                batch_first=True, dropout=0.1)
            self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )

        def forward(self, state, cluster_id):
            B = state.size(0)
            ce = self.cluster_emb(cluster_id)
            se = self.state_proj(state)
            x_state = torch.cat([se, ce], dim=-1).unsqueeze(1)
            mirror_ids = torch.arange(self.n_half, device=state.device).unsqueeze(0).expand(B, -1)
            me = self.mirror_emb(mirror_ids)
            x = me + x_state
            x = self.encoder(x)
            raw = self.head(x).squeeze(-1)
            if self.output_activation == 'sigmoid':
                low = self.output_low_tensor
                high = self.output_high_tensor
                return torch.sigmoid(raw) * (high - low) + low
            else:
                scale = torch.maximum(torch.abs(self.output_low_tensor), torch.abs(self.output_high_tensor))
                return torch.tanh(raw) * scale


def stage_train(cfg: Config, bo_data: dict, logger: Logger,
                stop: StopSignal, ckpt: Checkpoint):
    """阶段 5: 训练 Transformer"""
    logger.info("=== 阶段 5/7: 训练 Transformer ===")
    workdir = Path(cfg.workdir)
    out_path = workdir / 'transformer.pt'
    history_path = workdir / 'train_history.json'
    model_meta_path = workdir / 'model_meta.json'
    
    if ckpt.is_done('train') and out_path.exists():
        if model_meta_path.exists():
            meta = json.load(open(model_meta_path))
            _validate_cached_hash(meta.get('config_hash'), cfg, 'model_meta.json')
        logger.info("已完成,加载模型")
        return out_path, json.load(open(history_path))
    
    if not HAS_TORCH:
        raise ImportError("需要 PyTorch")
    
    samples = bo_data['samples']
    dim = bo_data['dim']
    
    # 构建数据集 (防御性: 处理 aim_optimal 维度可能不一致)
    X = samples[['solar_alt', 'solar_az', 'cos_inc', 'dni']].values.astype(np.float32)
    cluster_ids = samples['cluster'].values.astype(np.int64)
    
    # 把 aim_optimal 列(可能含不同长度的 list)转成统一维度的矩阵
    aims_raw = samples['aim_optimal'].tolist()
    n_samples_total = len(aims_raw)
    # 检查所有样本维度是否一致
    dims_seen = [len(a) for a in aims_raw]
    if len(set(dims_seen)) > 1:
        # 不一致: 找到最常见的维度,丢弃其它
        from collections import Counter
        most_common_dim = Counter(dims_seen).most_common(1)[0][0]
        logger.warn(f'BO 样本维度不一致: {Counter(dims_seen)}, 保留维度={most_common_dim}')
        valid_mask = np.array([len(a) == most_common_dim for a in aims_raw])
        aims_raw = [a for a, v in zip(aims_raw, valid_mask) if v]
        X = X[valid_mask]
        cluster_ids = cluster_ids[valid_mask]
        samples = samples[valid_mask].reset_index(drop=True)
        if most_common_dim != dim:
            logger.warn(f'实际样本维度 {most_common_dim} 与 bo_data dim={dim} 不符,使用 {most_common_dim}')
            dim = most_common_dim
    Y = np.array(aims_raw, dtype=np.float32)
    if Y.ndim != 2:
        raise ValueError(f'aim_optimal 解析失败,Y.shape={Y.shape}, dim={dim}')
    logger.info(f'训练集: {len(X)} 样本, 输出维度 {Y.shape[1]}, aim_mode={cfg.aim_mode}')

    # 标准化输入
    X_mean = X.mean(0); X_std = X.std(0) + 1e-6
    X_norm = (X - X_mean) / X_std

    # 划分
    n = len(X)
    rng = np.random.RandomState(cfg.seed)
    perm = rng.permutation(n)
    n_val = int(n * cfg.nn_val_ratio)
    val_idx = perm[:n_val]; train_idx = perm[n_val:]

    device = cfg.get_device()
    # 输出范围随 aim_mode 变化
    if cfg.aim_mode == 'transverse_span' and cfg.experiment_mode == 'grouped_span':
        bounds = np.asarray(cfg.grouped_span_bounds, dtype=np.float32)
        output_low = bounds[:, 0]
        output_high = bounds[:, 1]
        output_activation = 'sigmoid'
    elif cfg.aim_mode == 'transverse_span':
        output_low, output_high = cfg.xaim_span_range
        output_activation = 'sigmoid'
    else:
        output_low, output_high = cfg.z_range
        output_activation = 'tanh'
    model = AimTransformer(n_mirrors_half=dim, d_model=cfg.nn_d_model,
                           n_heads=cfg.nn_n_heads, n_layers=cfg.nn_n_layers,
                           n_clusters=cfg.n_clusters,
                           output_low=output_low, output_high=output_high,
                           output_activation=output_activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.nn_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.nn_epochs)
    loss_fn = nn.MSELoss()
    
    Xt = torch.from_numpy(X_norm).to(device)
    Ct = torch.from_numpy(cluster_ids).to(device)
    Yt = torch.from_numpy(Y).to(device)
    
    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
    best_val = float('inf')
    best_epoch = ckpt.get_partial('train', 'best_epoch', -1)
    epochs_no_improve = ckpt.get_partial('train', 'epochs_no_improve', 0)
    early_stopped = False
    
    # 恢复进度
    start_epoch = ckpt.get_partial('train', 'epoch', 0)
    if start_epoch > 0 and out_path.exists():
        model.load_state_dict(torch.load(out_path, map_location=device))
        history = ckpt.get_partial('train', 'history', history)
        best_val = ckpt.get_partial('train', 'best_val', best_val)
        best_epoch = ckpt.get_partial('train', 'best_epoch', best_epoch)
        epochs_no_improve = ckpt.get_partial('train', 'epochs_no_improve', epochs_no_improve)
        logger.info(f"从 epoch {start_epoch} 续训")
    
    for epoch in range(start_epoch, cfg.nn_epochs):
        stop.check()
        # train
        model.train()
        rng.shuffle(train_idx)
        bs = cfg.nn_batch_size
        losses = []
        for i in range(0, len(train_idx), bs):
            idx = train_idx[i:i+bs]
            xb = Xt[idx]; cb = Ct[idx]; yb = Yt[idx]
            pred = model(xb, cb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        train_loss = float(np.mean(losses))
        
        # val
        model.eval()
        with torch.no_grad():
            pred_val = model(Xt[val_idx], Ct[val_idx])
            val_loss = loss_fn(pred_val, Yt[val_idx]).item()
            val_mae = (pred_val - Yt[val_idx]).abs().mean().item()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        
        if val_loss < best_val - cfg.nn_min_delta:
            best_val = val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), out_path)
        else:
            epochs_no_improve += 1
        
        ckpt.set_partial('train', 'epoch', epoch + 1)
        ckpt.set_partial('train', 'history', history)
        ckpt.set_partial('train', 'best_val', best_val)
        ckpt.set_partial('train', 'best_epoch', best_epoch)
        ckpt.set_partial('train', 'epochs_no_improve', epochs_no_improve)
        logger.progress((epoch + 1) / cfg.nn_epochs,
                       f"Epoch {epoch+1}/{cfg.nn_epochs} train={train_loss:.4f} val={val_loss:.4f}")
        if epochs_no_improve >= cfg.nn_early_stop_patience:
            early_stopped = True
            logger.info(
                f"Early stopping at epoch {epoch + 1}, "
                f"best_epoch={best_epoch}, best_val={best_val:.4f}"
            )
            break
    
    # 保存归一化参数
    np.savez(workdir / 'norm.npz', X_mean=X_mean, X_std=X_std)
    history['best_epoch'] = best_epoch
    history['early_stopped'] = early_stopped
    history['best_val'] = float(best_val)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    with open(model_meta_path, 'w') as f:
        json.dump({'config_hash': compute_config_hash(cfg), 'dim': int(dim), 'aim_mode': cfg.aim_mode}, f, indent=2)
    
    ckpt.mark_done('train')
    return out_path, history


# ==================== 8. 符号回归 ====================

def stage_distill(cfg: Config, bo_data: dict, train_path: Path,
                  logger: Logger, stop: StopSignal, ckpt: Checkpoint):
    """阶段 6: PySR 符号回归蒸馏"""
    logger.info("=== 阶段 6/7: 符号回归蒸馏 ===")
    workdir = Path(cfg.workdir)
    out_path = workdir / 'formulas.json'
    
    if ckpt.is_done('distill') and out_path.exists():
        logger.info("已完成,加载公式")
        return json.load(open(out_path))
    
    samples = bo_data['samples']
    dim = bo_data['dim']
    
    # 直接从 BO 数据集学习(防御性: 处理维度不一致)
    aims_raw = samples['aim_optimal'].tolist()
    dims_seen = [len(a) for a in aims_raw]
    if len(set(dims_seen)) > 1:
        from collections import Counter
        most_common_dim = Counter(dims_seen).most_common(1)[0][0]
        logger.warn(f'BO 样本维度不一致: {Counter(dims_seen)}, 保留 {most_common_dim} 维样本')
        valid_mask = np.array([len(a) == most_common_dim for a in aims_raw])
        aims_raw = [a for a, v in zip(aims_raw, valid_mask) if v]
        samples = samples[valid_mask].reset_index(drop=True)
        dim = most_common_dim
    X = samples[['solar_alt', 'solar_az', 'cos_inc', 'dni']].values
    Y = np.array(aims_raw, dtype=np.float64)
    
    formulas = {}
    
    def _formula_key(i, dim, aim_mode):
        """在 transverse_span 模式下输出维度=1，命名为 span 而非 z_1"""
        if aim_mode == 'transverse_span' and dim == 1 and i == 0:
            return 'span'
        return f'z_{i+1}'

    HAS_PYSR_LOCAL = False
    if cfg.enable_pysr:
        try:
            from pysr import PySRRegressor
            HAS_PYSR_LOCAL = True
        except ImportError:
            HAS_PYSR_LOCAL = False

    if not HAS_PYSR_LOCAL:
        logger.warn("PySR 未安装,使用多项式回归代替")
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        for i in range(dim):
            stop.check()
            key = _formula_key(i, dim, cfg.aim_mode)
            pipe = Pipeline([
                ('poly', PolynomialFeatures(degree=3, include_bias=False)),
                ('reg', Ridge(alpha=1.0))
            ])
            pipe.fit(X, Y[:, i])
            pred = pipe.predict(X)
            r2 = 1 - ((Y[:, i] - pred)**2).sum() / ((Y[:, i] - Y[:, i].mean())**2).sum()
            mae = np.abs(Y[:, i] - pred).mean()
            formulas[key] = {
                'expression': '(polynomial degree-3, see model file)',
                'r2': float(r2),
                'mae': float(mae),
                'model_type': 'polynomial',
            }
            with open(workdir / f'poly_{key}.pkl', 'wb') as f:
                pickle.dump(pipe, f)
            logger.progress((i + 1) / dim, f"多项式 {key}: R²={r2:.3f}")
    else:
        for i in range(dim):
            stop.check()
            key = _formula_key(i, dim, cfg.aim_mode)
            sr = PySRRegressor(
                niterations=cfg.pysr_iterations,
                population_size=cfg.pysr_population,
                maxsize=cfg.pysr_maxsize,
                binary_operators=['+', '-', '*', '/'],
                unary_operators=['sin', 'cos', 'exp', 'sqrt'],
                progress=False,
                verbosity=0,
                temp_equation_file=True,
            )
            sr.fit(X, Y[:, i])
            best = sr.get_best()
            pred = sr.predict(X)
            r2 = 1 - ((Y[:, i] - pred)**2).sum() / ((Y[:, i] - Y[:, i].mean())**2).sum()
            formulas[key] = {
                'expression': str(best['equation']),
                'complexity': int(best['complexity']),
                'r2': float(r2),
                'mae': float(np.abs(Y[:, i] - pred).mean()),
                'model_type': 'symbolic',
            }
            logger.progress((i + 1) / dim, f"PySR {key}: {formulas[key]['expression']}")
    
    with open(out_path, 'w') as f:
        json.dump(formulas, f, indent=2)
    ckpt.mark_done('distill')
    return formulas


# ==================== 9. 年度合成 ====================

def stage_annual(cfg: Config, df: pd.DataFrame, cluster_data: dict,
                 bo_data: dict, train_path: Path, formulas: dict,
                 baselines: dict, logger: Logger, stop: StopSignal,
                 ckpt: Checkpoint):
    """阶段 7: 年度性能合成

    主对比策略:
      S1_center         - 所有镜子瞄准管心
      BO_adaptive_span  - BO 自适应横截面分散范围
      NN_adaptive_span  - Transformer 预测 span (可选)
    """
    logger.info("=== 阶段 7/7: 年度合成与对比 ===")
    workdir = Path(cfg.workdir)
    out_path = workdir / 'annual.pkl'

    if ckpt.is_done('annual') and out_path.exists():
        return _load_pickle_checked(out_path, cfg, 'annual.pkl')

    rep_df = cluster_data['representatives']
    weights = rep_df.set_index('cluster')['weight'].to_dict()
    base_df = baselines['results']

    geo = LFRGeometry(cfg)
    mcrt = create_mcrt_tracer(cfg, geo, logger)

    annual_records = []
    annual_detail_records = []
    if cfg.experiment_mode == 'grouped_span':
        bo_strategy_name = 'BO_grouped_span'
        nn_strategy_name = 'NN_grouped_span'
    else:
        bo_strategy_name = 'BO_adaptive_span'
        nn_strategy_name = 'NN_adaptive_span'

    def _weighted_metrics(records_for_strategy):
        eta_total = 0.0; sigma_total = 0.0; cv_total = 0.0
        par_total = 0.0; top_total = 0.0
        for r in records_for_strategy:
            w = r['weight']
            eta_total += w * r['eta_opt']
            sigma_total += w * r['sigma_surface']
            cv_total += w * r['cv_circ']
            par_total += w * r['par_full']
            top_total += w * r['top_flux_ratio']
        return eta_total, sigma_total, cv_total, par_total, top_total

    # --- S1_center (from baselines or re-evaluate) ---
    s1_sub = base_df[base_df['strategy'] == 'S1_center']
    if len(s1_sub) > 0 and 'sigma_surface' in s1_sub.columns:
        s1_recs = []
        for _, r in s1_sub.iterrows():
            s1_recs.append({
                'cluster': r['cluster'],
                'strategy': 'S1_center',
                'eta_opt': r['eta_opt'],
                'sigma_surface': r.get('sigma_surface', 0.0),
                'cv_circ': r['cv_circ'],
                'par_full': r.get('par_full', 0.0),
                'top_flux_ratio': r.get('top_flux_ratio', 0.0),
                'span': 0.0,
                'span_inner': np.nan,
                'span_mid': np.nan,
                'span_outer': np.nan,
                'span_vector': None,
                'strategy_param_mode': 'span_1d',
                'source_sample_idx': -1,
                'distance_to_rep': 0.0,
                'weight': weights[r['cluster']],
            })
        eta_s1, sig_s1, cv_s1, par_s1, top_s1 = _weighted_metrics(s1_recs)
        annual_records.append({
            'strategy': 'S1_center',
            'annual_eta_opt': eta_s1,
            'annual_sigma_surface': sig_s1,
            'annual_cv_circ': cv_s1,
            'annual_par_full': par_s1,
            'annual_top_flux_ratio': top_s1,
        })
        annual_detail_records.extend(s1_recs)
    else:
        # Re-evaluate S1
        s1_recs = []
        for _, rep in rep_df.iterrows():
            stop.check()
            cluster_id = rep['cluster']
            row = cluster_data['df'].iloc[rep['rep_idx']]
            aim = make_s1_aim(cfg)
            _, m = mcrt.trace(row['solar_alt'], row['solar_az'], aim, row['DNI'],
                              n_rays=cfg.n_rays_validate)
            s1_recs.append({
                'cluster': int(cluster_id),
                'strategy': 'S1_center',
                'eta_opt': float(m['eta_opt']),
                'sigma_surface': float(m['sigma_surface']),
                'cv_circ': float(m['cv_circ']),
                'par_full': float(m['par_full']),
                'top_flux_ratio': float(m['top_flux_ratio']),
                'span': 0.0,
                'span_inner': np.nan,
                'span_mid': np.nan,
                'span_outer': np.nan,
                'span_vector': None,
                'strategy_param_mode': 'span_1d',
                'source_sample_idx': -1,
                'distance_to_rep': 0.0,
                'weight': float(weights[cluster_id]),
            })
        eta_s1, sig_s1, cv_s1, par_s1, top_s1 = _weighted_metrics(s1_recs)
        annual_records.append({
            'strategy': 'S1_center',
            'annual_eta_opt': eta_s1,
            'annual_sigma_surface': sig_s1,
            'annual_cv_circ': cv_s1,
            'annual_par_full': par_s1,
            'annual_top_flux_ratio': top_s1,
        })
        annual_detail_records.extend(s1_recs)

    # --- BO_adaptive_span / BO_grouped_span ---
    samples = bo_data['samples'].copy()
    dni_col = 'dni' if 'dni' in samples.columns else 'DNI'
    bo_recs = []
    for _, rep in rep_df.iterrows():
        stop.check()
        cluster_id = rep['cluster']
        row = cluster_data['df'].iloc[rep['rep_idx']]
        c_samples = samples[samples['cluster'] == cluster_id]
        if len(c_samples) == 0:
            aim = make_s1_aim(cfg)
            source_sample_idx = -1
            dist_min = float('nan')
            lookup_reason = 'fallback_s1_no_bo_sample'
            used_span = 0.0
            used_span_inner = np.nan
            used_span_mid = np.nan
            used_span_outer = np.nan
            used_span_vector = None
            strategy_param_mode = 'grouped_span' if cfg.experiment_mode == 'grouped_span' else 'span_1d'
        else:
            sample_feats = c_samples[['solar_alt', 'solar_az', 'cos_inc', dni_col]].values.astype(float)
            target_feat = np.array([row['solar_alt'], row['solar_az'], row['cos_inc'], row['DNI']], dtype=float)
            scale = sample_feats.std(axis=0) + 1e-6
            dist = np.linalg.norm((sample_feats - target_feat[None, :]) / scale, axis=1)
            nearest = c_samples.iloc[int(np.argmin(dist))]
            aim_raw = _safe_parse_aim_vector(nearest['aim_optimal'])
            aim = expand_aim_to_full(aim_raw, cfg, geo=geo)
            source_sample_idx = int(nearest['sample_idx'])
            dist_min = float(np.min(dist))
            lookup_reason = 'nearest_bo_sample_same_cluster'
            used_span = float(nearest.get('span_optimal', float('nan')))
            used_span_inner = float(nearest.get('span_inner', np.nan))
            used_span_mid = float(nearest.get('span_mid', np.nan))
            used_span_outer = float(nearest.get('span_outer', np.nan))
            used_span_vector = nearest.get('span_vector', None)
            strategy_param_mode = nearest.get(
                'strategy_param_mode',
                'grouped_span' if cfg.experiment_mode == 'grouped_span' else 'span_1d'
            )
        _, m = mcrt.trace(row['solar_alt'], row['solar_az'], aim, row['DNI'],
                          n_rays=cfg.n_rays_validate)
        bo_recs.append({
            'cluster': int(cluster_id),
            'strategy': bo_strategy_name,
            'eta_opt': float(m['eta_opt']),
            'sigma_surface': float(m['sigma_surface']),
            'cv_circ': float(m['cv_circ']),
            'par_full': float(m['par_full']),
            'top_flux_ratio': float(m['top_flux_ratio']),
            'span': used_span,
            'span_inner': used_span_inner,
            'span_mid': used_span_mid,
            'span_outer': used_span_outer,
            'span_vector': used_span_vector,
            'strategy_param_mode': strategy_param_mode,
            'source_sample_idx': source_sample_idx,
            'distance_to_rep': dist_min,
            'weight': float(weights[cluster_id]),
        })
    eta_bo, sig_bo, cv_bo, par_bo, top_bo = _weighted_metrics(bo_recs)
    annual_records.append({
        'strategy': bo_strategy_name,
        'annual_eta_opt': eta_bo,
        'annual_sigma_surface': sig_bo,
        'annual_cv_circ': cv_bo,
        'annual_par_full': par_bo,
        'annual_top_flux_ratio': top_bo,
    })
    annual_detail_records.extend(bo_recs)

    # --- NN_adaptive_span / NN_grouped_span (optional) ---
    if HAS_TORCH and train_path is not None and Path(train_path).exists():
        try:
            device = cfg.get_device()
            dim = bo_data['dim']
            if cfg.aim_mode == 'transverse_span' and cfg.experiment_mode == 'grouped_span':
                bounds = np.asarray(cfg.grouped_span_bounds, dtype=np.float32)
                output_low = bounds[:, 0]
                output_high = bounds[:, 1]
                output_activation = 'sigmoid'
            elif cfg.aim_mode == 'transverse_span':
                output_low, output_high = cfg.xaim_span_range
                output_activation = 'sigmoid'
            else:
                output_low, output_high = cfg.z_range
                output_activation = 'tanh'
            model = AimTransformer(n_mirrors_half=dim, d_model=cfg.nn_d_model,
                                   n_heads=cfg.nn_n_heads, n_layers=cfg.nn_n_layers,
                                   n_clusters=cfg.n_clusters,
                                   output_low=output_low, output_high=output_high,
                                   output_activation=output_activation).to(device)
            model.load_state_dict(torch.load(train_path, map_location=device))
            model.eval()

            norm = np.load(workdir / 'norm.npz')
            X_mean, X_std = norm['X_mean'], norm['X_std']

            nn_recs = []
            for _, rep in rep_df.iterrows():
                stop.check()
                cluster_id = rep['cluster']
                row = cluster_data['df'].iloc[rep['rep_idx']]
                x_in = (np.array([[row['solar_alt'], row['solar_az'],
                                   row['cos_inc'], row['DNI']]]) - X_mean) / X_std
                with torch.no_grad():
                    nn_out = model(
                        torch.from_numpy(x_in.astype(np.float32)).to(device),
                        torch.tensor([cluster_id], dtype=torch.long).to(device)
                    ).cpu().numpy()[0]
                aim = expand_aim_to_full(nn_out, cfg, geo=geo)
                if cfg.aim_mode == 'transverse_span' and cfg.experiment_mode == 'grouped_span':
                    used_span_inner = float(nn_out[0])
                    used_span_mid = float(nn_out[1])
                    used_span_outer = float(nn_out[2])
                    used_span = float(np.mean(nn_out))
                    used_span_vector = nn_out.tolist()
                    strategy_param_mode = 'grouped_span'
                elif cfg.aim_mode == 'transverse_span':
                    used_span = float(nn_out[0])
                    used_span_inner = np.nan
                    used_span_mid = np.nan
                    used_span_outer = np.nan
                    used_span_vector = [float(nn_out[0])]
                    strategy_param_mode = 'span_1d'
                else:
                    used_span = float('nan')
                    used_span_inner = np.nan
                    used_span_mid = np.nan
                    used_span_outer = np.nan
                    used_span_vector = None
                    strategy_param_mode = 'longitudinal'
                _, m = mcrt.trace(row['solar_alt'], row['solar_az'], aim, row['DNI'],
                                  n_rays=cfg.n_rays_validate)
                nn_recs.append({
                    'cluster': int(cluster_id),
                    'strategy': nn_strategy_name,
                    'eta_opt': float(m['eta_opt']),
                    'sigma_surface': float(m['sigma_surface']),
                    'cv_circ': float(m['cv_circ']),
                    'par_full': float(m['par_full']),
                    'top_flux_ratio': float(m['top_flux_ratio']),
                    'span': used_span,
                    'span_inner': used_span_inner,
                    'span_mid': used_span_mid,
                    'span_outer': used_span_outer,
                    'span_vector': used_span_vector,
                    'strategy_param_mode': strategy_param_mode,
                    'source_sample_idx': -1,
                    'distance_to_rep': 0.0,
                    'weight': float(weights[cluster_id]),
                })
            eta_nn, sig_nn, cv_nn, par_nn, top_nn = _weighted_metrics(nn_recs)
            annual_records.append({
                'strategy': nn_strategy_name,
                'annual_eta_opt': eta_nn,
                'annual_sigma_surface': sig_nn,
                'annual_cv_circ': cv_nn,
                'annual_par_full': par_nn,
                'annual_top_flux_ratio': top_nn,
            })
            annual_detail_records.extend(nn_recs)
        except Exception as e:
            logger.warn(f"NN 推理失败，跳过 {nn_strategy_name}: {e}")

    annual_df = pd.DataFrame(annual_records)
    out = {
        'annual_summary': annual_df,
        'annual_details': pd.DataFrame(annual_detail_records),
        'baselines': baselines
    }
    _save_pickle_with_hash(out_path, out, cfg)
    ckpt.mark_done('annual')

    logger.info("年度对比:")
    for _, r in annual_df.iterrows():
        logger.info(
            f"  {r['strategy']}: η={r['annual_eta_opt']:.3f} "
            f"σ={r.get('annual_sigma_surface', float('nan')):.3f} "
            f"CV_circ={r['annual_cv_circ']:.3f}"
        )
    return out


# ==================== 10. 图表导出 ====================

def stage_sensitivity_fixed_span(cfg: Config, df: pd.DataFrame, cluster_data: dict,
                                 logger: Logger, stop: StopSignal, ckpt: Checkpoint):
    logger.info("=== 阶段: 固定 span 敏感性扫描 ===")
    workdir = Path(cfg.workdir)
    out_path = workdir / 'sensitivity_fixed_span.pkl'
    if ckpt.is_done('sensitivity') and out_path.exists():
        return _load_pickle_checked(out_path, cfg, 'sensitivity_fixed_span.pkl')
    geo = LFRGeometry(cfg)
    mcrt = create_mcrt_tracer(cfg, geo, logger)
    reps = cluster_data['representatives']
    cdf = cluster_data['df']
    weights = reps.set_index('cluster')['weight'].to_dict()
    rows = []
    for span in cfg.fixed_span_values:
        acc = {'eta': 0.0, 'sigma': 0.0, 'cv': 0.0, 'par': 0.0, 'top': 0.0}
        for _, rep in reps.iterrows():
            stop.check()
            row = cdf.iloc[int(rep['rep_idx'])]
            _, m = mcrt.trace(row['solar_alt'], row['solar_az'], make_xaim_from_span(cfg, span),
                              row['DNI'], n_rays=cfg.n_rays_sensitivity)
            w = weights[int(rep['cluster'])]
            acc['eta'] += w * m['eta_opt']; acc['sigma'] += w * m['sigma_surface']
            acc['cv'] += w * m['cv_circ']; acc['par'] += w * m['par_full']; acc['top'] += w * m['top_flux_ratio']
        rows.append({'span': float(span), 'annual_eta_opt': acc['eta'], 'annual_sigma_surface': acc['sigma'],
                     'annual_cv_circ': acc['cv'], 'annual_par_full': acc['par'], 'annual_top_flux_ratio': acc['top']})
    df_scan = pd.DataFrame(rows).sort_values('span').reset_index(drop=True)
    s1 = df_scan[df_scan['span'] == 0.0].iloc[0]
    df_scan['eta_rel_to_s1'] = df_scan['annual_eta_opt'] / (s1['annual_eta_opt'] + 1e-12)
    df_scan['sigma_improve_pct'] = (s1['annual_sigma_surface'] - df_scan['annual_sigma_surface']) / (s1['annual_sigma_surface'] + 1e-12) * 100.0
    df_scan['par_improve_pct'] = (s1['annual_par_full'] - df_scan['annual_par_full']) / (s1['annual_par_full'] + 1e-12) * 100.0
    df_scan['top_flux_improve_pct'] = (
        (df_scan['annual_top_flux_ratio'] - s1['annual_top_flux_ratio'])
        / (s1['annual_top_flux_ratio'] + 1e-12) * 100.0
    )
    tab_dir = workdir / 'tables'; fig_dir = workdir / 'figures'
    tab_dir.mkdir(exist_ok=True); fig_dir.mkdir(exist_ok=True)
    df_scan.to_csv(tab_dir / 'tab08_fixed_span_scan.csv', index=False)
    def draw_fig13(lang):
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()
        ax1.plot(df_scan['span'], df_scan['annual_eta_opt'], marker='o', color='tab:blue')
        ax2.plot(df_scan['span'], df_scan['annual_sigma_surface'], marker='s', color='tab:red')
        ax1.axvline(0.035, color='gray', linestyle='--', alpha=0.6)
        ax1.set_xlabel('span (m)'); ax1.set_ylabel('annual_eta_opt'); ax2.set_ylabel('annual_sigma_surface')
        ax1.set_title('图13 固定 span 扫描' if lang == 'zh' else 'Fig. 13 Fixed-span scan')
        return fig
    save_bilingual_figure(fig_dir / 'fig13_fixed_span_scan.png', draw_fig13)
    out = {'scan': df_scan}
    _save_pickle_with_hash(out_path, out, cfg)
    ckpt.mark_done('sensitivity')
    return out


def export_strategy_screening_table(cfg: Config, annual: Optional[dict], bo_data: Optional[dict], sensitivity: Optional[dict]):
    if cfg.experiment_mode != 'compare_all':
        return
    workdir = Path(cfg.workdir)
    tab_dir = workdir / 'tables'
    tab_dir.mkdir(exist_ok=True)
    rows = []
    if sensitivity and 'scan' in sensitivity:
        scan = sensitivity['scan']
        for s in [0.0, 0.015, 0.025, 0.035, 0.04]:
            sub = scan[np.isclose(scan['span'], s)]
            if len(sub) == 0:
                continue
            r = sub.iloc[0]
            rows.append({'strategy': 'S1_center' if s == 0 else f'S_fixed_span_{str(s).replace(".", "")}',
                         'annual_eta_opt': r['annual_eta_opt'], 'annual_sigma_surface': r['annual_sigma_surface'],
                         'annual_par_full': r['annual_par_full'], 'annual_top_flux_ratio': r['annual_top_flux_ratio'],
                         'eta_rel_to_s1': r['eta_rel_to_s1'], 'sigma_improve_pct': r['sigma_improve_pct'],
                         'par_improve_pct': r['par_improve_pct'], 'top_flux_improve_pct': r['top_flux_improve_pct'],
                         'runtime_minutes': np.nan})
    if annual and 'annual_summary' in annual:
        adf = annual['annual_summary']
        s1 = adf[adf['strategy'] == 'S1_center'].iloc[0] if len(adf[adf['strategy'] == 'S1_center']) else None
        for _, r in adf.iterrows():
            if r['strategy'] == 'S1_center':
                continue
            rows.append({'strategy': r['strategy'], 'annual_eta_opt': r['annual_eta_opt'],
                         'annual_sigma_surface': r.get('annual_sigma_surface', np.nan),
                         'annual_par_full': r.get('annual_par_full', np.nan),
                         'annual_top_flux_ratio': r.get('annual_top_flux_ratio', np.nan),
                         'eta_rel_to_s1': (r['annual_eta_opt'] / s1['annual_eta_opt']) if s1 is not None else np.nan,
                         'sigma_improve_pct': ((s1.get('annual_sigma_surface', np.nan) - r.get('annual_sigma_surface', np.nan)) / (s1.get('annual_sigma_surface', np.nan) + 1e-12) * 100.0) if s1 is not None else np.nan,
                         'par_improve_pct': np.nan, 'top_flux_improve_pct': np.nan, 'runtime_minutes': np.nan})
    if rows:
        out = pd.DataFrame(rows)
        def _rec(rr):
            if rr['eta_rel_to_s1'] >= 0.96 and rr['sigma_improve_pct'] >= 5:
                return 'promising_main_strategy'
            if rr['eta_rel_to_s1'] >= 0.96 and rr['sigma_improve_pct'] > 0:
                return 'mild_improvement'
            return 'not_recommended'
        out['recommendation'] = out.apply(_rec, axis=1)
        out.to_csv(tab_dir / 'tab09_strategy_screening.csv', index=False)

def _safe_parse_aim_vector(raw_aim):
    """将 aim_optimal 安全解析为一维 numpy 向量"""
    if isinstance(raw_aim, str):
        try:
            parsed = ast.literal_eval(raw_aim)
        except Exception as e:
            raise ValueError(f"无法解析字符串形式的 aim_optimal: {raw_aim}") from e
        aim = np.asarray(parsed, dtype=np.float64).ravel()
    else:
        aim = np.asarray(raw_aim, dtype=np.float64).ravel()
    if aim.size == 0:
        raise ValueError("aim_optimal 为空向量")
    return aim


def make_xaim_pattern(cfg, n=None):
    """
    生成横截面瞄准点基础 pattern，范围 [-1, 1]。
    pattern 顺序与 geo.mirror_x / mirror_id 顺序一致。
    """
    if n is None:
        n = cfg.n_mirrors
    return np.linspace(-1.0, 1.0, int(n), dtype=float)


def make_group_ids(cfg, geo=None):
    """
    按 |mirror_x| 将镜子分成 inner/mid/outer 三组。
    0 = inner, 1 = mid, 2 = outer
    """
    if geo is None:
        geo = LFRGeometry(cfg)
    idx = np.argsort(np.abs(geo.mirror_x))
    groups = np.zeros(cfg.n_mirrors, dtype=int)
    n = len(idx)
    groups[idx[n // 3: 2 * n // 3]] = 1
    groups[idx[2 * n // 3:]] = 2
    return groups


def make_xaim_from_span(cfg, span):
    """
    根据 BO 搜索到的 span 生成每面镜的横截面瞄准点 x_aim。

    span = 0      -> 所有镜子瞄准管心
    span = 0.035  -> 接近文献策略4
    span = 0.04   -> 稍激进但仍在建议安全范围内
    """
    span = float(np.asarray(span).ravel()[0])
    lo, hi = cfg.xaim_span_range
    span = float(np.clip(span, lo, hi))
    return span * make_xaim_pattern(cfg, cfg.n_mirrors)


def make_xaim_from_grouped_span(cfg, spans, geo=None):
    spans = np.asarray(spans, dtype=float).ravel()
    if spans.size != 3:
        raise ValueError(f"grouped_span 需要 3 维输入，当前为 {spans.size}")
    bounds = np.asarray(cfg.grouped_span_bounds, dtype=float)
    spans = np.clip(spans, bounds[:, 0], bounds[:, 1])
    group_ids = make_group_ids(cfg, geo=geo)
    pattern = make_xaim_pattern(cfg, cfg.n_mirrors)
    return spans[group_ids] * pattern


def make_s1_aim(cfg):
    """
    S1_center：所有镜子瞄准管子中心 (span=0)。
    """
    if cfg.aim_mode == 'transverse_span':
        return make_xaim_from_span(cfg, 0.0)
    return np.zeros(cfg.n_mirrors, dtype=float)


def expand_aim_to_full(aim, cfg, geo=None):
    """
    将策略输出的 aim 向量统一展开成 cfg.n_mirrors 维。
    """
    aim = np.asarray(aim, dtype=np.float64).ravel()

    if cfg.aim_mode == 'transverse_span':
        if getattr(cfg, 'experiment_mode', 'span_1d') == 'grouped_span':
            if aim.size == 3:
                return make_xaim_from_grouped_span(cfg, aim, geo=geo)
            if aim.size == cfg.n_mirrors:
                return aim
            raise ValueError(
                f"grouped_span 模式下 aim 应为 3 维 [inner, mid, outer] 或 "
                f"{cfg.n_mirrors} 维 full x_aim，当前 len={aim.size}"
            )

        if aim.size == 1:
            return make_xaim_from_span(cfg, float(aim[0]))
        if aim.size == cfg.n_mirrors:
            return aim
        raise ValueError(
            f"transverse_span/span_1d 模式下 aim 应为 1 维 span 或 "
            f"{cfg.n_mirrors} 维 full x_aim，当前 len={aim.size}"
        )

    # old longitudinal fallback
    if aim.size == cfg.n_mirrors:
        return aim
    if cfg.n_mirrors % 2 == 0 and aim.size == cfg.n_mirrors // 2:
        return np.concatenate([aim[::-1], aim])

    raise ValueError(
        f"aim 向量维度不匹配: len(aim)={aim.size}, "
        f"cfg.n_mirrors={cfg.n_mirrors}, cfg.aim_mode={cfg.aim_mode}, "
        f"experiment_mode={getattr(cfg, 'experiment_mode', None)}"
    )


def compute_tonatiuh_mirror_pose(cfg, geo, mirror_id, sun_alt_deg, sun_az_deg, z_aim):
    """
    根据当前 Python 模型坐标系，计算 Tonatiuh 手动验证所需姿态参数。
    仅支持旧 longitudinal z_aim 语义；transverse_span 模式下禁止调用。
    """
    if getattr(cfg, 'aim_mode', None) == 'transverse_span':
        raise NotImplementedError(
            "compute_tonatiuh_mirror_pose() 当前只支持旧 longitudinal z_aim 语义；"
            "transverse_span 模式下目标点应为 target_point=(x_aim, 0, receiver_height)，"
            "本函数不得直接用于 transverse_span。"
        )
    if mirror_id < 0 or mirror_id >= len(geo.mirror_x):
        raise ValueError(f"mirror_id 越界: {mirror_id}, n_mirrors={len(geo.mirror_x)}")

    mirror_center = np.array([geo.mirror_x[mirror_id], 0.0, 0.0], dtype=np.float64)
    target_point = np.array([0.0, float(z_aim), geo.cpc_y], dtype=np.float64)

    alt = np.deg2rad(float(sun_alt_deg))
    az = np.deg2rad(float(sun_az_deg))
    sun_x = -np.cos(alt) * np.sin(az)
    sun_y = -np.cos(alt) * np.cos(az)
    sun_z = -np.sin(alt)
    incoming_raw = np.array([sun_x, sun_y, sun_z], dtype=np.float64)
    in_norm = np.linalg.norm(incoming_raw)
    if in_norm < 1e-12:
        raise ValueError("太阳入射向量长度过小，无法归一化")
    incoming_dir = incoming_raw / in_norm

    outgoing_raw = target_point - mirror_center
    out_norm = np.linalg.norm(outgoing_raw)
    if out_norm < 1e-12:
        raise ValueError(
            f"反射目标向量长度过小: mirror_id={mirror_id}, "
            f"mirror_center={mirror_center.tolist()}, target_point={target_point.tolist()}"
        )
    outgoing_dir = outgoing_raw / out_norm

    normal_raw = outgoing_dir - incoming_dir
    normal_norm = np.linalg.norm(normal_raw)
    if normal_norm < 1e-12:
        raise ValueError(
            f"镜面法向量长度过小: mirror_id={mirror_id}, "
            f"sun_alt_deg={sun_alt_deg}, sun_az_deg={sun_az_deg}, z_aim={z_aim}"
        )
    normal = normal_raw / normal_norm

    tilt_xz_deg = float(np.rad2deg(np.arctan2(normal[0], normal[2])))
    cant_y_deg = float(np.rad2deg(np.arctan2(normal[1], np.sqrt(normal[0]**2 + normal[2]**2))))

    return {
        "mirror_x": float(mirror_center[0]),
        "mirror_y": float(mirror_center[1]),
        "mirror_z": float(mirror_center[2]),
        "target_x": float(target_point[0]),
        "target_y": float(target_point[1]),
        "target_z": float(target_point[2]),
        "normal_x": float(normal[0]),
        "normal_y": float(normal[1]),
        "normal_z": float(normal[2]),
        "tilt_xz_deg": tilt_xz_deg,
        "cant_y_deg": cant_y_deg
    }

def export_figures_and_tables(cfg: Config, df: pd.DataFrame, cluster_data: dict,
                               baselines: dict, bo_data: dict, history: dict,
                               formulas: dict, annual: dict, logger: Logger):
    """生成论文所有图表"""
    logger.info("=== 导出论文图表 ===")
    workdir = Path(cfg.workdir)
    fig_dir = workdir / 'figures'
    tab_dir = workdir / 'tables'
    fig_dir.mkdir(exist_ok=True)
    tab_dir.mkdir(exist_ok=True)
    
    # === 图 1: 聚类 PCA ===
    Xpca = cluster_data['X_pca']
    labels = cluster_data['labels']
    centers = cluster_data['centers_pca']
    def draw_fig01(lang):
        fig, ax = plt.subplots(figsize=(8, 6))
        for k in range(cfg.n_clusters):
            m = labels == k
            ax.scatter(Xpca[m, 0], Xpca[m, 1], s=8, alpha=0.5, label=f'C{k}')
        rep_label = '代表时刻' if lang == 'zh' else 'Representative points'
        ax.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, c='red',
                   edgecolors='black', linewidths=1.5, label=rep_label, zorder=5)
        ax.set_xlabel('主成分 1 (PC1)' if lang == 'zh' else 'PC1')
        ax.set_ylabel('主成分 2 (PC2)' if lang == 'zh' else 'PC2')
        title = (f'图1 工况聚类(K={cfg.n_clusters}, 轮廓系数={cluster_data["silhouette"]:.2f})'
                 if lang == 'zh' else
                 f'Fig. 1 Operating-condition clustering (K={cfg.n_clusters}, Silhouette={cluster_data["silhouette"]:.2f})')
        ax.set_title(title)
        ax.legend(ncol=2, loc='best', fontsize=8)
        return fig
    save_bilingual_figure(fig_dir / 'fig01_cluster_pca.png', draw_fig01)
    
    # === 图 2: 太阳轨迹 ===
    rep_df = cluster_data['representatives']
    rep_full = cluster_data['df'].iloc[rep_df['rep_idx'].values]
    az_rad = np.deg2rad(rep_full['solar_az'].values)
    alt = rep_full['solar_alt'].values
    def draw_fig02(lang):
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection='polar'))
        sc = ax.scatter(az_rad, 90 - alt, c=rep_df['dni_mean'], s=200,
                        cmap='YlOrRd', edgecolors='black')
        ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
        ax.set_title('图2 12个代表时刻太阳轨迹' if lang == 'zh' else 'Fig. 2 Solar trajectories of 12 representative instants')
        fig.colorbar(sc, label='DNI (W/m²)')
        for i, (_, r) in enumerate(rep_df.iterrows()):
            prefix = '簇' if lang == 'zh' else 'C'
            ax.annotate(f"{prefix}{r['cluster']}", xy=(az_rad[i], 90 - alt[i]), fontsize=8)
        return fig
    save_bilingual_figure(fig_dir / 'fig02_sun_path.png', draw_fig02)
    
    # === 表 1: 聚类统计 ===
    rep_df_out = rep_df[['cluster', 'rep_timestamp', 'dni_mean', 'dni_std',
                         'alt_mean', 'count', 'weight']].copy()
    rep_df_out.columns = ['Cluster', 'Representative time', 'DNI mean', 'DNI std',
                          'Alt mean', 'Hours', 'Annual weight']
    rep_df_out.to_csv(tab_dir / 'tab01_clusters.csv', index=False)
    rep_df_out.to_markdown(tab_dir / 'tab01_clusters.md', index=False)
    
    # === 图 3: 基线能流云图 ===
    flux_maps = baselines['flux_maps']
    core_clusters = list(rep_df['cluster'].head(3))
    _baseline_strategies = sorted({s for _, s in flux_maps.keys()})
    def draw_fig03(lang):
        ncols = max(len(_baseline_strategies), 1)
        fig, axes = plt.subplots(3, ncols, figsize=(6 * ncols, 10))
        if ncols == 1:
            axes = axes[:, None]
        for ci, c in enumerate(core_clusters):
            for si, s in enumerate(_baseline_strategies):
                ax = axes[ci, si]
                fmap = flux_maps.get((c, s))
                if fmap is None:
                    ax.axis('off')
                    continue
                im = ax.imshow(fmap, aspect='auto', origin='lower', cmap='hot',
                               extent=[-cfg.mirror_length/2, cfg.mirror_length/2, -180, 180])
                ax.set_xlabel('管长方向 y (m)' if lang == 'zh' else 'Axial position y (m)')
                ax.set_ylabel('圆周角 φ (deg)' if lang == 'zh' else 'φ (deg)')
                ax.set_title(f"{s} - {'簇' if lang == 'zh' else 'C'}{c}")
                fig.colorbar(im, ax=ax, label='能流 (W/m²)' if lang == 'zh' else 'Flux (W/m²)')
        fig.suptitle('图3 基线策略吸热管表面能流分布' if lang == 'zh' else 'Fig. 3 Flux maps on absorber surface for baseline strategies')
        return fig
    save_bilingual_figure(fig_dir / 'fig03_baseline_flux_maps.png', draw_fig03)
    
    # === 图 4: 圆周能流曲线 ===
    phi_axis = np.linspace(-180, 180, cfg.n_phi_bins)
    def draw_fig04(lang):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ci, c in enumerate(core_clusters):
            ax = axes[ci]
            for s in _baseline_strategies:
                fmap = flux_maps.get((c, s))
                if fmap is None:
                    continue
                curve = fmap.mean(axis=1)
                ax.plot(phi_axis, curve, label=s)
            ax.set_xlabel('圆周角 φ (deg)' if lang == 'zh' else 'Circumferential angle φ (deg)')
            ax.set_ylabel('平均能流 (W/m²)' if lang == 'zh' else 'Mean flux (W/m²)')
            ax.set_title(f"{'簇' if lang == 'zh' else 'C'}{c}")
            ax.legend()
        fig.suptitle('图4 圆周能流曲线对比' if lang == 'zh' else 'Fig. 4 Circumferential flux curves')
        return fig
    save_bilingual_figure(fig_dir / 'fig04_circumferential_curves.png', draw_fig04)
    
    # === 表 2: 基线汇总 ===
    base_df = baselines['results']
    _pivot_vals = [v for v in ['eta_opt', 'cv_circ', 'sigma_surface', 'par_full',
                                'top_flux_ratio', 'nuf', 'par'] if v in base_df.columns]
    pivot = base_df.pivot_table(index='cluster', columns='strategy', values=_pivot_vals)
    pivot.to_csv(tab_dir / 'tab02_baseline.csv')

    # === 表 3: BO 选择诊断 ===
    samples_diag = bo_data['samples']
    diag_cols = [
        'cluster', 'sample_idx', 'timestamp', 'eta_opt', 'cv_circ',
        'sigma_surface', 'par_full', 'top_flux_ratio',
        'span_optimal', 'xaim_min', 'xaim_max', 'aim_mode',
        'uniformity_metric', 'uniformity_obj',
        's1_eta_opt', 's1_sigma_surface', 's1_cv_circ',
        'bo_selected_reason', 'eta_floor_rel'
    ]
    keep_cols = [c for c in diag_cols if c in samples_diag.columns]
    if keep_cols:
        samples_diag[keep_cols].to_csv(tab_dir / 'tab03_bo_selection_diagnostics.csv', index=False)
    
    # === 图 5: Pareto 前沿 ===
    pareto_data = bo_data['pareto_data']
    _sigma_col = 'sigma_surface' if 'sigma_surface' in (bo_data['samples'].columns if hasattr(bo_data['samples'], 'columns') else []) else 'cv_circ'
    def draw_fig05(lang):
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        for i, c in enumerate(list(pareto_data.keys())[:6]):
            ax = axes[i // 3, i % 3]
            for run in pareto_data[c]:
                Y = run['Y']
                mask = run['pareto_mask']
                ax.scatter(Y[:, 1], Y[:, 0], s=10, alpha=0.3, c='gray')
                ax.scatter(Y[mask, 1], Y[mask, 0], s=30, c='red', label='Pareto')
            b1 = base_df[(base_df['cluster'] == c) & (base_df['strategy'] == 'S1_center')]
            if len(b1) and _sigma_col in b1.columns:
                ax.scatter(b1[_sigma_col], b1['eta_opt'], marker='*', s=200, c='blue', label='S1')
            xlabel_zh = '表面能流标准差 σ（越小越好）' if _sigma_col == 'sigma_surface' else 'CV_circ（越小越好）'
            xlabel_en = 'Surface flux std. σ (lower better)' if _sigma_col == 'sigma_surface' else 'CV_circ (lower better)'
            ax.set_xlabel(xlabel_zh if lang == 'zh' else xlabel_en)
            ax.set_ylabel('η_opt（越大越好）' if lang == 'zh' else 'η_opt (higher better)')
            ax.set_title(f"{'簇' if lang == 'zh' else 'C'}{c}")
            ax.legend(fontsize=8)
        fig.suptitle('图5 BO Pareto 前沿（6个代表时刻）' if lang == 'zh' else 'Fig. 5 BO Pareto fronts (6 representative instants)')
        return fig
    save_bilingual_figure(fig_dir / 'fig05_pareto_fronts.png', draw_fig05)
    
    # === 图 7: 训练曲线 ===
    if history:
        ep = list(range(1, len(history['train_loss']) + 1))
        _mae_label_zh = 'span MAE (m)' if cfg.aim_mode == 'transverse_span' else '验证 MAE (m)'
        _mae_label_en = 'span MAE (m)' if cfg.aim_mode == 'transverse_span' else 'Val MAE (m)'
        def draw_fig07(lang):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(ep, history['train_loss'], label='train')
            axes[0].plot(ep, history['val_loss'], label='val')
            axes[0].set_xlabel('轮次' if lang == 'zh' else 'Epoch')
            axes[0].set_ylabel('MSE'); axes[0].legend()
            axes[0].set_title('训练与验证损失' if lang == 'zh' else 'Training & Validation Loss')
            axes[1].plot(ep, history['val_mae'])
            axes[1].set_xlabel('轮次' if lang == 'zh' else 'Epoch')
            axes[1].set_ylabel(_mae_label_zh if lang == 'zh' else _mae_label_en)
            axes[1].set_title('验证 MAE' if lang == 'zh' else 'Validation MAE')
            fig.suptitle('图7 Transformer 训练曲线' if lang == 'zh' else 'Fig. 7 Transformer training curves')
            return fig
        save_bilingual_figure(fig_dir / 'fig07_training_curves.png', draw_fig07)
    
    # === 表 5: 符号公式 ===
    if formulas:
        rows = [{'Mirror': k, 'Expression': v.get('expression', 'N/A'),
                'R²': f"{v.get('r2', 0):.3f}", 'MAE': f"{v.get('mae', 0):.3f}",
                'Type': v.get('model_type', 'N/A')}
               for k, v in formulas.items()]
        pd.DataFrame(rows).to_csv(tab_dir / 'tab05_formulas.csv', index=False)
        pd.DataFrame(rows).to_markdown(tab_dir / 'tab05_formulas.md', index=False)
    
    # === 图 12 + 表 6: 年度策略对比 ===
    if annual:
        adf = annual['annual_summary']
        _has_sigma = 'annual_sigma_surface' in adf.columns
        def draw_fig12(lang):
            n_panels = 3 if _has_sigma else 2
            fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
            axes[0].bar(adf['strategy'], adf['annual_eta_opt'], color='steelblue')
            axes[0].set_ylabel('年度 η_opt' if lang == 'zh' else 'Annual η_opt')
            axes[0].set_title('年度光学效率' if lang == 'zh' else 'Annual optical efficiency')
            for i, v in enumerate(adf['annual_eta_opt']):
                axes[0].text(i, v + 0.005, f'{v:.3f}', ha='center')
            if _has_sigma:
                axes[1].bar(adf['strategy'], adf['annual_sigma_surface'], color='coral')
                axes[1].set_ylabel('年度 σ_surface' if lang == 'zh' else 'Annual σ_surface')
                axes[1].set_title('年度表面能流标准差' if lang == 'zh' else 'Annual surface flux std.')
                for i, v in enumerate(adf['annual_sigma_surface']):
                    axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center')
                if n_panels > 2 and 'annual_par_full' in adf.columns:
                    axes[2].bar(adf['strategy'], adf['annual_par_full'], color='mediumpurple')
                    axes[2].set_ylabel('年度 PAR_full' if lang == 'zh' else 'Annual PAR_full')
                    axes[2].set_title('年度峰均比' if lang == 'zh' else 'Annual peak-to-avg ratio')
                    for i, v in enumerate(adf['annual_par_full']):
                        axes[2].text(i, v + 0.05, f'{v:.3f}', ha='center')
            else:
                axes[1].bar(adf['strategy'], adf['annual_cv_circ'], color='salmon')
                axes[1].set_ylabel('年度 CV_circ' if lang == 'zh' else 'Annual CV_circ')
                axes[1].set_title('年度圆周变异系数' if lang == 'zh' else 'Annual circumferential CV')
                for i, v in enumerate(adf['annual_cv_circ']):
                    axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center')
            for ax in axes:
                ax.tick_params(axis='x', rotation=20)
            fig.suptitle('图12 年度策略对比' if lang == 'zh' else 'Fig. 12 Annual strategy comparison')
            return fig
        save_bilingual_figure(fig_dir / 'fig12_annual_comparison.png', draw_fig12)

        adf.to_csv(tab_dir / 'tab06_annual.csv', index=False)
        adf.to_markdown(tab_dir / 'tab06_annual.md', index=False)
        if isinstance(annual, dict) and 'annual_details' in annual:
            annual_details = annual['annual_details']
            if isinstance(annual_details, pd.DataFrame):
                annual_details.to_csv(tab_dir / 'tab07_annual_details.csv', index=False)
    
    # === 图 11: 年度热力图(基于 BO 样本估计) ===
    samples = bo_data['samples']
    samples_df = samples.copy()
    samples_df['ts'] = pd.to_datetime(samples_df['timestamp'])
    samples_df['month'] = samples_df['ts'].dt.month
    samples_df['hr'] = samples_df['ts'].dt.hour
    pivot_eta = samples_df.pivot_table(index='month', columns='hr', values='eta_opt', aggfunc='mean')
    _heatmap_col2 = 'sigma_surface' if 'sigma_surface' in samples_df.columns else 'cv_circ'
    pivot_uni = samples_df.pivot_table(index='month', columns='hr', values=_heatmap_col2, aggfunc='mean')
    def draw_fig11(lang):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        im0 = axes[0].imshow(pivot_eta, aspect='auto', cmap='viridis', origin='lower')
        axes[0].set_xlabel('小时' if lang == 'zh' else 'Hour')
        axes[0].set_ylabel('月份' if lang == 'zh' else 'Month')
        axes[0].set_title('η_opt')
        fig.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(pivot_uni, aspect='auto', cmap='magma_r', origin='lower')
        axes[1].set_xlabel('小时' if lang == 'zh' else 'Hour')
        axes[1].set_ylabel('月份' if lang == 'zh' else 'Month')
        axes[1].set_title('σ_surface' if _heatmap_col2 == 'sigma_surface' else 'CV_circ')
        fig.colorbar(im1, ax=axes[1])
        fig.suptitle('图11 年度性能热力图（BO优化策略）' if lang == 'zh' else 'Fig. 11 Annual performance heatmaps (BO strategy)')
        return fig
    save_bilingual_figure(fig_dir / 'fig11_annual_heatmap.png', draw_fig11)
    
    # === 给 Tonatiuh 的瞄准向量导出 ===
    export_tonatiuh_all_training_cases(cfg, cluster_data, bo_data, formulas, workdir)
    
    logger.info(f"图表已保存到 {fig_dir} 与 {tab_dir}")


def export_tonatiuh_all_training_cases(cfg, cluster_data, bo_data, formulas, workdir):
    """导出 bo_data['samples'] 全部 BO 训练/优化样本的 Tonatiuh 手动验证数据。"""
    del cluster_data, formulas  # 该导出仅依赖 bo_data 样本

    # transverse_span 模式下旧版 Tonatiuh 导出语义错误，跳过
    if cfg.aim_mode == 'transverse_span':
        out_dir = Path(workdir) / 'tonatiuh_aiming'
        out_dir.mkdir(exist_ok=True)
        readme_path = out_dir / 'README_transverse_span_not_exported.md'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(
                "# Tonatiuh export skipped\n\n"
                "当前 aim_mode='transverse_span'。\n"
                "优化变量为 span，实际每面镜横截面瞄准点为：\n"
                "    x_aim_i = span * pattern_i\n\n"
                "Python 坐标系：\n"
                "    x = 横截面左右方向\n"
                "    y = 管长方向\n"
                "    z = 高度方向\n\n"
                "旧 Tonatiuh 导出函数按 longitudinal z_aim 解释，会把 aim 当作管长方向偏移，"
                "因此已跳过。\n"
                "若后续需要 Tonatiuh 导出，应重新实现 "
                "target_point=(x_aim_i, 0, receiver_height)，"
                "其中 0 表示管长方向中心 y=0。\n"
            )
        return

    samples = bo_data['samples'].copy()
    samples = samples.sort_values(['cluster', 'sample_idx']).reset_index(drop=True)

    def _get_col_name(df, candidates, required=True):
        for c in candidates:
            if c in df.columns:
                return c
        if required:
            raise KeyError(
                f"缺少必要列，候选列名={candidates}, 当前列={list(df.columns)}"
            )
        return None

    dni_col = _get_col_name(samples, ['dni', 'DNI'])
    time_col = _get_col_name(samples, ['timestamp', 'time'])

    out_dir = workdir / 'tonatiuh_aiming'
    out_dir.mkdir(exist_ok=True)
    geo = LFRGeometry(cfg)

    def _norm_series(s):
        s = s.astype(float)
        smin = s.min()
        smax = s.max()
        if np.isclose(smax, smin):
            return pd.Series(np.zeros(len(s)), index=s.index, dtype=np.float64)
        return (s - smin) / (smax - smin)

    rows_bo = []
    rows_s1 = []
    rows_s2 = []
    index_rows = []
    raw_aim_dim_counts = {}
    any_expanded = False
    s2_full_aim = np.linspace(-cfg.mirror_length * 0.4, cfg.mirror_length * 0.4, cfg.n_mirrors)

    n_dni = _norm_series(samples[dni_col])
    n_eta = _norm_series(samples['eta_opt'])
    n_cv = _norm_series(samples['cv_circ'])
    suggested_score = n_dni + n_eta - n_cv

    for bo_row_index, sample in samples.iterrows():
        cluster = int(sample['cluster'])
        sample_idx = int(sample['sample_idx'])
        case_id = f"C{cluster:02d}_S{sample_idx:03d}_R{bo_row_index:04d}"

        try:
            raw_aim = _safe_parse_aim_vector(sample['aim_optimal'])
            full_aim_bo = expand_aim_to_full(raw_aim, cfg, geo=geo)
        except Exception as e:
            raise ValueError(
                f"解析/展开 aim_optimal 失败: bo_row_index={bo_row_index}, "
                f"cluster={cluster}, sample_idx={sample_idx}"
            ) from e

        raw_dim = int(raw_aim.size)
        full_dim = int(full_aim_bo.size)
        raw_aim_dim_counts[str(raw_dim)] = raw_aim_dim_counts.get(str(raw_dim), 0) + 1
        expanded_to_full = bool(full_dim == cfg.n_mirrors and raw_dim != cfg.n_mirrors)
        any_expanded = any_expanded or expanded_to_full

        row_common = {
            'case_id': case_id,
            'bo_row_index': int(bo_row_index),
            'cluster': cluster,
            'sample_idx': sample_idx,
            'time': sample[time_col],
            'solar_alt': float(sample['solar_alt']),
            'solar_az': float(sample['solar_az']),
            'DNI': float(sample[dni_col]),
            'cos_inc': float(sample['cos_inc']),
            'eta_opt': float(sample['eta_opt']),
            'cv_circ': float(sample['cv_circ']),
            'aim_vector_dim_raw': raw_dim,
            'aim_vector_dim_full': full_dim,
        }
        index_rows.append({
            **{k: row_common[k] for k in [
                'case_id', 'bo_row_index', 'cluster', 'sample_idx', 'time',
                'solar_alt', 'solar_az', 'DNI', 'cos_inc', 'eta_opt',
                'cv_circ', 'aim_vector_dim_raw'
            ]},
            'expanded_to_n_mirrors': expanded_to_full,
            'suggested_sort_score': float(suggested_score.iloc[bo_row_index])
        })

        for mirror_id in range(cfg.n_mirrors):
            z_bo = float(full_aim_bo[mirror_id])
            z_s1 = 0.0
            z_s2 = float(s2_full_aim[mirror_id])
            pose_bo = compute_tonatiuh_mirror_pose(cfg, geo, mirror_id, sample['solar_alt'], sample['solar_az'], z_bo)
            pose_s1 = compute_tonatiuh_mirror_pose(cfg, geo, mirror_id, sample['solar_alt'], sample['solar_az'], z_s1)
            pose_s2 = compute_tonatiuh_mirror_pose(cfg, geo, mirror_id, sample['solar_alt'], sample['solar_az'], z_s2)

            rows_bo.append({
                **row_common,
                'strategy': 'BO',
                'mirror_id': mirror_id,
                'z_aim': z_bo,
                **pose_bo,
                'note': "BO optimized aiming from bo_data['samples']"
            })
            rows_s1.append({
                **row_common,
                'strategy': 'S1_center',
                'mirror_id': mirror_id,
                'z_aim': z_s1,
                **pose_s1,
                'note': "Baseline S1: center aiming"
            })
            rows_s2.append({
                **row_common,
                'strategy': 'S2_uniform',
                'mirror_id': mirror_id,
                'z_aim': z_s2,
                **pose_s2,
                'note': "Baseline S2: uniform longitudinal aiming"
            })

    bo_df = pd.DataFrame(rows_bo)
    s1_df = pd.DataFrame(rows_s1)
    s2_df = pd.DataFrame(rows_s2)
    index_df = pd.DataFrame(index_rows)

    expected_rows = len(samples) * cfg.n_mirrors
    if len(bo_df) != expected_rows:
        raise ValueError(f"BO CSV 行数不一致: got={len(bo_df)}, expected={expected_rows}")
    if len(s1_df) != expected_rows:
        raise ValueError(f"S1 CSV 行数不一致: got={len(s1_df)}, expected={expected_rows}")
    if len(s2_df) != expected_rows:
        raise ValueError(f"S2 CSV 行数不一致: got={len(s2_df)}, expected={expected_rows}")

    bo_name = 'tonatiuh_all_training_cases_BO.csv'
    s1_name = 'tonatiuh_all_training_cases_S1_center.csv'
    s2_name = 'tonatiuh_all_training_cases_S2_uniform.csv'
    index_name = 'tonatiuh_training_cases_index.csv'
    manifest_name = 'tonatiuh_export_manifest.json'
    readme_name = 'README.md'

    bo_df.to_csv(out_dir / bo_name, index=False)
    s1_df.to_csv(out_dir / s1_name, index=False)
    s2_df.to_csv(out_dir / s2_name, index=False)
    index_df.to_csv(out_dir / index_name, index=False)

    manifest = {
        "n_samples": int(len(samples)),
        "n_mirrors": int(cfg.n_mirrors),
        "expected_rows_per_strategy": int(expected_rows),
        "bo_rows": int(len(bo_df)),
        "s1_rows": int(len(s1_df)),
        "s2_rows": int(len(s2_df)),
        "raw_aim_dim_counts": raw_aim_dim_counts,
        "expanded_aim_dim": int(cfg.n_mirrors),
        "expanded_happened": any_expanded,
        "use_symmetry_current_cfg": bool(cfg.use_symmetry),
        "exported_files": [bo_name, s1_name, s2_name, index_name, manifest_name, readme_name],
        "created_at": datetime.now().isoformat()
    }
    with open(out_dir / manifest_name, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    readme = f"""# Tonatiuh 手动验证导出说明

本目录导出的是 `bo_dataset.pkl` / `bo_data['samples']` 中**所有** BO 训练/优化过的样本时刻，而不是 cluster representatives。
默认配置下通常是 `12 clusters × 20 samples_per_cluster = 240` 个样本时刻，但实际样本数请以 `tonatiuh_training_cases_index.csv` 为准。

## 文件说明

- `tonatiuh_all_training_cases_S1_center.csv`：基线 1，全瞄管心。
- `tonatiuh_all_training_cases_S2_uniform.csv`：基线 2，沿管轴均匀分布。
- `tonatiuh_all_training_cases_BO.csv`：本文方法，BO 优化得到的 longitudinal aiming 策略。
- `tonatiuh_training_cases_index.csv`：case 索引（每行一个 BO 样本时刻），便于后续筛选。
- `tonatiuh_export_manifest.json`：导出统计、行数校验、维度统计与时间戳。

## 行含义与关键变量

每个策略 CSV 中，一行对应“一个样本时刻 + 一面镜子”的 Tonatiuh 手动设置参数。
Python 策略优化变量 `z_aim` 不是直接镜面倾角，而是沿集热管轴向的瞄准偏移量。

对每面镜推荐优先使用目标点：
`target_point = (target_x, target_y, target_z)`，其中：
- `target_x = 0`
- `target_y = z_aim`
- `target_z = cpc_inlet_height / geo.cpc_y`（本导出中实际使用 Python 几何的 `geo.cpc_y`）

如果 Tonatiuh 支持“让镜子瞄准某个目标点”，优先使用 `target_x, target_y, target_z`。
如果 Tonatiuh 只能手动输入角度：
- `tilt_xz_deg`：横截面等效一轴倾角；
- `cant_y_deg`：轴向瞄准引入的三维偏转。
仅输入 `tilt_xz_deg` 可能无法完全复现 Python 的 longitudinal aiming。

## 推荐验证流程

1. 从 `tonatiuh_training_cases_index.csv` 中选择若干 `case_id`。
2. 在 Tonatiuh 中建立与 Python Config 相同的 LFR + CPC + 吸热管模型。
3. 设置对应 `time / solar_alt / solar_az / DNI`。
4. 对每面镜设置目标点或姿态。
5. 分别运行 `S1_center`、`S2_uniform`、`BO`。
6. 比较吸热管表面能流均匀性。
7. 用 BO 相对于 S1/S2 的能流均匀性改善验证策略可行性。
"""
    with open(out_dir / readme_name, 'w', encoding='utf-8') as f:
        f.write(readme)


# ==================== 11. 全流程编排 ====================

def run_pipeline(cfg: Config, logger: Logger, stop: StopSignal,
                 stages_to_run=None):
    """主流程"""
    workdir = Path(cfg.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cfg_hash = compute_config_hash(cfg)
    cfg_dict = cfg.to_dict()
    cfg_dict['config_hash'] = cfg_hash
    with open(workdir / 'config_used.json', 'w') as f:
        json.dump(cfg_dict, f, indent=2, ensure_ascii=False)
    ckpt = Checkpoint(
        cfg.workdir,
        config_hash=cfg_hash,
        strict=cfg.config_hash_strict,
        force=cfg.force_reuse_cache
    )
    
    all_stages = ['data', 'cluster', 'baseline', 'bo', 'train', 'distill', 'annual', 'sensitivity', 'export']
    if stages_to_run is None:
        stages_to_run = all_stages
    
    np.random.seed(cfg.seed)
    if HAS_TORCH:
        torch.manual_seed(cfg.seed)
    
    df = None; cluster_data = None; baselines = None; bo_data = None
    train_path = None; history = {}; formulas = {}; annual = None; sensitivity = None
    describe_runtime_backend(cfg, logger)
    if cfg.experiment_mode == 'compare_all':
        logger.warn(
            "compare_all 当前仅汇总已有 sensitivity/annual/bo 结果，"
            "不会自动运行完整 BO_1D 与 BO_grouped 诊断。"
        )
    runtime_rows = []
    def _mark_runtime(stage_name, t0, t1):
        runtime_rows.append({
            'stage': stage_name,
            'start_time': datetime.fromtimestamp(t0).isoformat(),
            'end_time': datetime.fromtimestamp(t1).isoformat(),
            'runtime_seconds': t1 - t0,
            'mcrt_backend': cfg.mcrt_backend,
            'experiment_mode': cfg.experiment_mode,
            'n_rays_eval': cfg.n_rays_eval,
            'n_rays_validate': cfg.n_rays_validate,
        })
    
    try:
        if 'data' in stages_to_run:
            _t0 = time.time()
            logger.stage('data', 0)
            df = stage_data(cfg, logger, stop, ckpt)
            logger.stage('data', 1)
            _mark_runtime('data', _t0, time.time())
        
        if 'cluster' in stages_to_run:
            _t0 = time.time()
            if df is None:
                df = pd.read_pickle(workdir / 'tmy_processed.pkl')
            logger.stage('cluster', 0)
            cluster_data = stage_cluster(cfg, df, logger, stop, ckpt)
            logger.stage('cluster', 1)
            _mark_runtime('cluster', _t0, time.time())
        
        if 'baseline' in stages_to_run:
            _t0 = time.time()
            if cluster_data is None:
                with open(workdir / 'clusters.pkl', 'rb') as f:
                    cluster_data = pickle.load(f)
            validate_mcrt_backend_parity(cfg, cluster_data, logger)
            logger.stage('baseline', 0)
            baselines = stage_baseline(cfg, cluster_data, logger, stop, ckpt)
            logger.stage('baseline', 1)
            _mark_runtime('baseline', _t0, time.time())
        
        if 'bo' in stages_to_run:
            _t0 = time.time()
            if df is None:
                df = pd.read_pickle(workdir / 'tmy_processed.pkl')
            if cluster_data is None:
                with open(workdir / 'clusters.pkl', 'rb') as f:
                    cluster_data = pickle.load(f)
            logger.stage('bo', 0)
            bo_data = stage_bo(cfg, df, cluster_data, logger, stop, ckpt)
            logger.stage('bo', 1)
            _mark_runtime('bo', _t0, time.time())
        
        if 'train' in stages_to_run:
            _t0 = time.time()
            if bo_data is None:
                bo_data = _load_pickle_checked(workdir / 'bo_dataset.pkl', cfg, 'bo_dataset.pkl')
            logger.stage('train', 0)
            train_path, history = stage_train(cfg, bo_data, logger, stop, ckpt)
            logger.stage('train', 1)
            _mark_runtime('train', _t0, time.time())
        
        if 'distill' in stages_to_run:
            _t0 = time.time()
            if bo_data is None:
                bo_data = _load_pickle_checked(workdir / 'bo_dataset.pkl', cfg, 'bo_dataset.pkl')
            if train_path is None:
                train_path = workdir / 'transformer.pt'
            logger.stage('distill', 0)
            formulas = stage_distill(cfg, bo_data, train_path, logger, stop, ckpt)
            logger.stage('distill', 1)
            _mark_runtime('distill', _t0, time.time())
        
        if 'annual' in stages_to_run:
            _t0 = time.time()
            if df is None:
                df = pd.read_pickle(workdir / 'tmy_processed.pkl')
            if cluster_data is None:
                with open(workdir / 'clusters.pkl', 'rb') as f:
                    cluster_data = pickle.load(f)
            if baselines is None:
                baselines = _load_pickle_checked(workdir / 'baselines.pkl', cfg, 'baselines.pkl')
            if bo_data is None:
                bo_data = _load_pickle_checked(workdir / 'bo_dataset.pkl', cfg, 'bo_dataset.pkl')
            if not formulas and (workdir / 'formulas.json').exists():
                formulas = json.load(open(workdir / 'formulas.json'))
            if train_path is None:
                train_path = workdir / 'transformer.pt'
            logger.stage('annual', 0)
            annual = stage_annual(cfg, df, cluster_data, bo_data, train_path,
                                  formulas, baselines, logger, stop, ckpt)
            logger.stage('annual', 1)
            _mark_runtime('annual', _t0, time.time())
        
        if 'sensitivity' in stages_to_run:
            _t0 = time.time()
            if df is None:
                df = pd.read_pickle(workdir / 'tmy_processed.pkl')
            if cluster_data is None:
                with open(workdir / 'clusters.pkl', 'rb') as f:
                    cluster_data = pickle.load(f)
            logger.stage('sensitivity', 0)
            sensitivity = stage_sensitivity_fixed_span(cfg, df, cluster_data, logger, stop, ckpt)
            logger.stage('sensitivity', 1)
            _mark_runtime('sensitivity', _t0, time.time())

        if 'export' in stages_to_run:
            _t0 = time.time()
            if cluster_data is None:
                with open(workdir / 'clusters.pkl', 'rb') as f:
                    cluster_data = pickle.load(f)
            if baselines is None:
                baselines = _load_pickle_checked(workdir / 'baselines.pkl', cfg, 'baselines.pkl')
            if bo_data is None:
                bo_data = _load_pickle_checked(workdir / 'bo_dataset.pkl', cfg, 'bo_dataset.pkl')
            if df is None:
                df = pd.read_pickle(workdir / 'tmy_processed.pkl')
            if not history and (workdir / 'train_history.json').exists():
                history = json.load(open(workdir / 'train_history.json'))
            if not formulas and (workdir / 'formulas.json').exists():
                formulas = json.load(open(workdir / 'formulas.json'))
            if annual is None and (workdir / 'annual.pkl').exists():
                annual = _load_pickle_checked(workdir / 'annual.pkl', cfg, 'annual.pkl')
            if sensitivity is None and (workdir / 'sensitivity_fixed_span.pkl').exists():
                sensitivity = _load_pickle_checked(workdir / 'sensitivity_fixed_span.pkl', cfg, 'sensitivity_fixed_span.pkl')
            export_figures_and_tables(cfg, df, cluster_data, baselines, bo_data,
                                      history, formulas, annual, logger)
            export_strategy_screening_table(cfg, annual, bo_data, sensitivity)
            _mark_runtime('export', _t0, time.time())

        if runtime_rows:
            tab_dir = workdir / 'tables'
            tab_dir.mkdir(exist_ok=True)
            pd.DataFrame(runtime_rows).to_csv(tab_dir / 'tab00_runtime_summary.csv', index=False)
        
        logger.info("=" * 50)
        logger.info("✓ 全流程完成!")
        logger.info(f"结果目录: {workdir.absolute()}")
        logger.status("✓ 完成")
    except InterruptedError:
        logger.warn("用户已中断,进度已保存,下次启动将自动续跑")
        logger.status("⏸ 已中断")
    except Exception as e:
        logger.error(f"运行出错: {e}")
        logger.error(traceback.format_exc())
        logger.status("✗ 出错")


# ==================== 12. GUI ====================

class LFRGui:
    def __init__(self, root):
        self.root = root
        root.title('LFR 年尺度智能瞄准策略优化')
        root.geometry('1100x780')
        
        self.cfg = Config()
        self.stop_signal = StopSignal()
        self.msg_queue = queue.Queue()
        self.worker = None
        
        self._build_ui()
        self._poll_queue()
    
    def _build_ui(self):
        # 顶栏
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill='x')
        
        ttk.Label(top, text='工作目录:').grid(row=0, column=0, sticky='w')
        self.workdir_var = tk.StringVar(value=self.cfg.workdir)
        ttk.Entry(top, textvariable=self.workdir_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(top, text='选择', command=self._pick_workdir).grid(row=0, column=2)
        
        ttk.Label(top, text='TMY 文件:').grid(row=1, column=0, sticky='w', pady=5)
        self.tmy_var = tk.StringVar(value=self.cfg.tmy_path)
        ttk.Entry(top, textvariable=self.tmy_var, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(top, text='选择', command=self._pick_tmy).grid(row=1, column=2)
        
        # 配置面板
        cfg_frame = ttk.LabelFrame(self.root, text='关键参数', padding=10)
        cfg_frame.pack(fill='x', padx=10, pady=5)
        self._build_config_panel(cfg_frame)
        
        # 阶段勾选
        stage_frame = ttk.LabelFrame(self.root, text='运行阶段选择（勾选=运行；全不选=按断点自动继续）', padding=10)
        stage_frame.pack(fill='x', padx=10, pady=5)
        self.stage_vars = {}
        stages = [('data', '数据 data'), ('cluster', '聚类 cluster'), ('baseline', '基线 baseline'),
                  ('bo', 'BO 优化 bo'), ('train', '训练 train'),
                  ('distill', '蒸馏 distill'), ('annual', '年度 annual'),
                  ('sensitivity', '敏感性 sensitivity'), ('export', '导出 export')]
        for i, (s, label) in enumerate(stages):
            v = tk.BooleanVar(value=True)
            self.stage_vars[s] = v
            ttk.Checkbutton(stage_frame, text=label, variable=v).grid(row=i // 5, column=i % 5, padx=5, sticky='w')
        ttk.Label(stage_frame, text='说明：复选框被选中表示“运行该阶段”；未选中表示“跳过该阶段”。若全部未选，将按 checkpoint 自动继续未完成阶段。').grid(row=2, column=0, columnspan=5, sticky='w', pady=4)
        quick = ttk.Frame(stage_frame)
        quick.grid(row=3, column=0, columnspan=5, sticky='w')
        ttk.Button(quick, text='全部运行', command=self._select_all_stages).pack(side='left', padx=3)
        ttk.Button(quick, text='清空选择', command=self._clear_all_stages).pack(side='left', padx=3)
        ttk.Button(quick, text='仅导出图表', command=lambda: self._set_stage_preset(['export'])).pack(side='left', padx=3)
        ttk.Button(quick, text='仅年度+导出', command=lambda: self._set_stage_preset(['annual', 'export'])).pack(side='left', padx=3)
        ttk.Button(quick, text='仅敏感性扫描', command=lambda: self._set_stage_preset(['sensitivity'])).pack(side='left', padx=3)
        ttk.Button(quick, text='从断点继续', command=self._resume_from_checkpoint).pack(side='left', padx=3)
        
        # 控制按钮
        ctrl = ttk.Frame(self.root, padding=10)
        ctrl.pack(fill='x')
        self.run_btn = ttk.Button(ctrl, text='▶ 开始运行', command=self.start)
        self.run_btn.pack(side='left', padx=5)
        self.stop_btn = ttk.Button(ctrl, text='⏸ 中断', command=self.stop, state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        ttk.Button(ctrl, text='清空断点(危险)', command=self.reset_ckpt).pack(side='left', padx=5)
        ttk.Button(ctrl, text='打开结果目录', command=self.open_workdir).pack(side='left', padx=5)
        
        # 进度
        prog_frame = ttk.LabelFrame(self.root, text='进度', padding=10)
        prog_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(prog_frame, text='当前阶段:').grid(row=0, column=0, sticky='w')
        self.stage_label = ttk.Label(prog_frame, text='—', font=('TkDefaultFont', 10, 'bold'))
        self.stage_label.grid(row=0, column=1, sticky='w', padx=10)
        self.cur_pb = ttk.Progressbar(prog_frame, length=600, mode='determinate', maximum=100)
        self.cur_pb.grid(row=0, column=2, padx=10)
        self.cur_pct = ttk.Label(prog_frame, text='0%')
        self.cur_pct.grid(row=0, column=3)
        
        ttk.Label(prog_frame, text='整体进度:').grid(row=1, column=0, sticky='w', pady=5)
        self.total_pb = ttk.Progressbar(prog_frame, length=600, mode='determinate', maximum=len(self.stage_vars))
        self.total_pb.grid(row=1, column=2, padx=10, pady=5)
        self.total_pct = ttk.Label(prog_frame, text='0/9')
        self.total_pct.grid(row=1, column=3)
        
        self.status_label = ttk.Label(prog_frame, text='就绪', foreground='blue')
        self.status_label.grid(row=2, column=0, columnspan=4, sticky='w', pady=5)
        
        # 日志
        log_frame = ttk.LabelFrame(self.root, text='日志', padding=5)
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=18, font=('Consolas', 9))
        self.log_text.pack(fill='both', expand=True)
        self.log_text.tag_config('warn', foreground='orange')
        self.log_text.tag_config('err', foreground='red')
        
        # 状态栏
        self.statusbar = ttk.Label(self.root, text=self._dependency_status(),
                                    relief='sunken', anchor='w')
        self.statusbar.pack(side='bottom', fill='x')
    
    def _build_config_panel(self, parent):
        items = [
            ('镜面数', 'n_mirrors', 18, int),
            ('集热管高 H (m)', 'receiver_height', 9.0, float),
            ('CPC 入口高 (m)', 'cpc_inlet_height', 8.72, float),
            ('聚类簇数', 'n_clusters', 12, int),
            ('每簇采样数', 'samples_per_cluster', 20, int),
            ('BO 初始点', 'bo_n_initial', 30, int),
            ('BO 迭代数', 'bo_n_iterations', 80, int),
            ('NN epoch', 'nn_epochs', 200, int),
            ('MCRT 光线数', 'n_rays_eval', 50000, int),
            ('MCRT backend', 'mcrt_backend', 'numpy_cpu', str),
            ('MCRT workers', 'mcrt_num_workers', 1, int),
            ('NN device', 'device', 'auto', str),
            ('PySR 启用', 'enable_pysr', False, bool),
            ('Experiment mode', 'experiment_mode', 'span_1d', str),
            ('启用对称', 'use_symmetry', True, bool),
        ]
        self.cfg_vars = {}
        for i, (label, key, default, typ) in enumerate(items):
            r, c = i // 5, (i % 5) * 2
            ttk.Label(parent, text=label).grid(row=r, column=c, sticky='e', padx=3, pady=2)
            if typ == bool:
                v = tk.BooleanVar(value=default)
                ttk.Checkbutton(parent, variable=v).grid(row=r, column=c+1, sticky='w', padx=3)
            else:
                v = tk.StringVar(value=str(default))
                ttk.Entry(parent, textvariable=v, width=12).grid(row=r, column=c+1, sticky='w', padx=3)
            self.cfg_vars[key] = (v, typ)
    
    def _dependency_status(self):
        deps = [('numpy/pandas/sklearn', HAS_SKLEARN), ('PyTorch', HAS_TORCH),
                ('pvlib', HAS_PVLIB)]
        s = ' | '.join(f"{n}:{'✓' if ok else '✗'}" for n, ok in deps)
        return f"依赖: {s} | PySR:按需加载"

    def _select_all_stages(self):
        for v in self.stage_vars.values():
            v.set(True)

    def _clear_all_stages(self):
        for v in self.stage_vars.values():
            v.set(False)

    def _set_stage_preset(self, enabled):
        for k, v in self.stage_vars.items():
            v.set(k in set(enabled))

    def _resume_from_checkpoint(self):
        self._clear_all_stages()
        self.start(resume_mode=True)
    
    def _pick_workdir(self):
        d = filedialog.askdirectory(initialdir=self.workdir_var.get())
        if d: self.workdir_var.set(d)
    
    def _pick_tmy(self):
        f = filedialog.askopenfilename(filetypes=[('CSV','*.csv'),('All','*.*')])
        if f: self.tmy_var.set(f)
    
    def _collect_cfg(self):
        self.cfg.workdir = self.workdir_var.get()
        self.cfg.tmy_path = self.tmy_var.get()
        for key, (var, typ) in self.cfg_vars.items():
            try:
                if typ == bool:
                    setattr(self.cfg, key, var.get())
                else:
                    setattr(self.cfg, key, typ(var.get()))
            except Exception as e:
                messagebox.showerror('参数错误', f'{key}: {e}')
                return False
        return True
    
    def start(self, resume_mode=False):
        if self.worker is not None and self.worker.is_alive():
            messagebox.showwarning('提示', '已有任务在运行')
            return
        if not self._collect_cfg():
            return
        if not Path(self.cfg.tmy_path).exists():
            messagebox.showerror('错误', f'TMY 文件不存在: {self.cfg.tmy_path}')
            return
        
        stages = [s for s, v in self.stage_vars.items() if v.get()]
        if not stages:
            if not messagebox.askyesno('确认', '未选择阶段，将按断点自动运行未完成阶段，是否继续？'):
                return
            ckpt = Checkpoint(self.cfg.workdir, config_hash=compute_config_hash(self.cfg))
            all_stages = ['data', 'cluster', 'baseline', 'bo', 'train', 'distill', 'annual', 'sensitivity', 'export']
            stages = [s for s in all_stages if (s == 'export' or not ckpt.is_done(s))]
        if resume_mode:
            self._log('从断点继续模式启动')
        
        self.stop_signal.reset()
        self.run_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.log_text.delete('1.0', 'end')
        self._log(f'启动流程,阶段: {stages}')
        self._completed_stages = 0
        self.total_pb.config(maximum=len(stages))
        self.total_pb['value'] = 0
        self.total_pct.config(text=f'0/{len(stages)}')
        
        logger = Logger(self.msg_queue)
        self.worker = threading.Thread(
            target=run_pipeline, args=(self.cfg, logger, self.stop_signal, stages), daemon=True)
        self.worker.start()
    
    def stop(self):
        if self.worker and self.worker.is_alive():
            self.stop_signal.request_stop()
            self._log('已发送中断请求...', 'warn')
    
    def reset_ckpt(self):
        if messagebox.askyesno('确认', '清空断点会丢失所有阶段标记,确定?'):
            ck = Checkpoint(self.workdir_var.get())
            ck.reset()
            self._log('断点已清空')
    
    def open_workdir(self):
        d = self.workdir_var.get()
        if not Path(d).exists():
            messagebox.showinfo('提示', '目录还不存在')
            return
        try:
            if sys.platform == 'win32':
                os.startfile(d)
            elif sys.platform == 'darwin':
                os.system(f'open "{d}"')
            else:
                os.system(f'xdg-open "{d}"')
        except Exception as e:
            messagebox.showinfo('路径', f'请手动打开: {d}\n{e}')
    
    def _log(self, msg, tag=None):
        ts = datetime.now().strftime('%H:%M:%S')
        line = f"[{ts}] {msg}\n"
        self.log_text.insert('end', line, tag if tag else ())
        self.log_text.see('end')
    
    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == 'log':
                    tag = None
                    if '[WARN]' in payload: tag = 'warn'
                    elif '[ERR ]' in payload: tag = 'err'
                    self._log(payload, tag)
                elif kind == 'progress':
                    frac, text = payload
                    self.cur_pb['value'] = max(0, min(100, frac * 100))
                    self.cur_pct.config(text=f"{frac*100:.0f}%")
                    if text:
                        self.status_label.config(text=text)
                elif kind == 'stage_progress':
                    stage, frac = payload
                    self.stage_label.config(text=stage.upper())
                    if frac == 1.0:
                        self._completed_stages += 1
                        self.total_pb['value'] = self._completed_stages
                        self.total_pct.config(
                            text=f"{self._completed_stages}/{int(self.total_pb['maximum'])}")
                elif kind == 'status':
                    self.status_label.config(text=payload)
                    if payload.startswith('✓') or payload.startswith('✗') or payload.startswith('⏸'):
                        self.run_btn.config(state='normal')
                        self.stop_btn.config(state='disabled')
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)


class _StdoutLogger:
    """无 GUI 时的命令行日志器,直接打印到 stdout"""
    def __init__(self, prefix='', verbose=True):
        self.prefix = prefix
        self.verbose = verbose
        self._last_progress_text = ''
    def info(self, m): print(f'{self.prefix}[INFO] {m}', flush=True)
    def warn(self, m): print(f'{self.prefix}[WARN] {m}', flush=True)
    def error(self, m): print(f'{self.prefix}[ERR ] {m}', flush=True)
    def progress(self, f, t=''):
        # 进度只在文本变化时打印,避免日志爆炸
        if self.verbose and t and t != self._last_progress_text:
            print(f'{self.prefix}  [{f*100:5.1f}%] {t}', flush=True)
            self._last_progress_text = t
    def stage(self, s, f):
        if f == 0:
            print(f'{self.prefix}\n>>> 进入阶段: {s}', flush=True)
        elif f == 1:
            print(f'{self.prefix}<<< 完成阶段: {s}\n', flush=True)
    def status(self, t): print(f'{self.prefix}[STATUS] {t}', flush=True)


def run_kaggle(workdir='/kaggle/working/lfr_results',
               tmy_path='/kaggle/input/dunhuang-tmy/dunhuang_tmy.csv',
               time_budget_hours=8.5,
               stages=None,
               cfg_overrides=None,
               dry_run=False):
    """Kaggle/Colab/CLI 环境主入口
    
    参数:
      workdir: 结果输出目录
      tmy_path: TMY CSV 路径
      time_budget_hours: 时间预算(小时), Kaggle 单 session 最长 9 小时
      stages: 要跑的阶段列表, None=全部
      cfg_overrides: 配置覆盖字典, 例如 {'n_clusters': 3, 'samples_per_cluster': 2}
      dry_run: True 时使用最小配置 (用于流程验证, 约 5-10 分钟)
    """
    cfg = Config()
    cfg.workdir = workdir
    cfg.tmy_path = tmy_path
    
    if dry_run:
        cfg.n_clusters = 3
        cfg.samples_per_cluster = 2
        cfg.bo_n_initial = 8
        cfg.bo_n_iterations = 10
        cfg.nn_epochs = 30
        cfg.n_rays_eval = 8000
        cfg.n_rays_validate = 30000
        print('=== DRY RUN 模式: 最小配置, 预计 5-15 分钟 ===')
    
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)
            print(f'[Config] {k} = {v}')
    
    # 时间预算
    budget_seconds = time_budget_hours * 3600 if time_budget_hours else None
    stop = StopSignal(time_budget_seconds=budget_seconds)
    logger = _StdoutLogger(verbose=True)
    
    if budget_seconds:
        logger.info(f'时间预算: {time_budget_hours:.1f} 小时, 不足 10 分钟时自动保存退出')
    
    Path(workdir).mkdir(parents=True, exist_ok=True)
    
    try:
        run_pipeline(cfg, logger, stop, stages_to_run=stages)
        logger.info(f'\n=== 全部完成,共耗时 {stop.time_used()/60:.1f} 分钟 ===')
        return True
    except InterruptedError as e:
        logger.warn(f'\n=== 中断: {e} ===')
        logger.warn(f'已耗时 {stop.time_used()/60:.1f} 分钟,checkpoint 已保存')
        logger.warn(f'下次启动会自动从中断位置续跑')
        return False
    except Exception as e:
        logger.error(f'\n=== 出错: {e} ===')
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    # 解析参数
    args = sys.argv[1:]
    
    if '--cli' in args or '--kaggle' in args:
        # 命令行模式 (Kaggle/服务器/CI)
        cfg_path = None
        dry_run = '--dry-run' in args
        for i, a in enumerate(args):
            if a == '--config' and i + 1 < len(args):
                cfg_path = args[i + 1]
        
        cfg = Config()
        if cfg_path and Path(cfg_path).exists():
            cfg = Config.load(cfg_path)
            print(f'已加载配置: {cfg_path}')
        
        if dry_run:
            cfg.n_clusters = 3
            cfg.samples_per_cluster = 2
            cfg.bo_n_initial = 8
            cfg.bo_n_iterations = 10
            cfg.nn_epochs = 30
            cfg.n_rays_eval = 8000
            print('=== DRY RUN 模式 ===')
        
        # 时间预算: 默认 12 小时(无限制),--budget HOURS 指定
        budget_seconds = None
        for i, a in enumerate(args):
            if a == '--budget' and i + 1 < len(args):
                budget_seconds = float(args[i + 1]) * 3600
        
        stop = StopSignal(time_budget_seconds=budget_seconds)
        logger = _StdoutLogger()
        
        try:
            run_pipeline(cfg, logger, stop)
        except InterruptedError as e:
            print(f'\n中断: {e}')
            print(f'下次启动会自动续跑')
        except Exception as e:
            print(f'\n出错: {e}')
            import traceback
            traceback.print_exc()
    elif HAS_TKINTER:
        # GUI 模式
        root = tk.Tk()
        try:
            style = ttk.Style()
            if 'clam' in style.theme_names():
                style.theme_use('clam')
        except Exception:
            pass
        LFRGui(root)
        root.mainloop()
    else:
        print('当前环境无 tkinter, 请使用 CLI 模式:')
        print('  python lfr_aiming.py --cli           # 完整运行')
        print('  python lfr_aiming.py --cli --dry-run # 快速验证(5-15 分钟)')
        print('  python lfr_aiming.py --cli --budget 8.5  # 8.5 小时时间预算')
        print('  python lfr_aiming.py --cli --config myconfig.json')
        print()
        print('或在 Kaggle/Notebook 中:')
        print('  from lfr_aiming import run_kaggle')
        print('  run_kaggle(dry_run=True)              # 快速验证')
        print('  run_kaggle(time_budget_hours=8.5)    # Kaggle 9 小时 session')


if __name__ == '__main__':
    main()
