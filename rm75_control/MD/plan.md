# RM75-6F 8-DOF 可达性图 (Capability Map) + 滑轨底座优化 —— 详细实施计划

> 目标：离线为 RM75-6F 生成 Zacharias-2013 风格的 Capability Map，然后按 Vahrenkamp-2013 的
> Reachability-Inversion 思路，在线求解 (a) 单条扫描轨迹的最优滑轨基座 y_b 与 (b) 在无法整段扫描时
> 求出「从起点开始能连续完成的最长前缀」。全流程只依赖 URDF + Pinocchio + numpy/scipy + PyVista，
> **不引入 Genesis**。可视化直接读取 `assets/robots/rm75_6f_8dof/meshes/*.dae`。

---

## 0. 输入 / 约束回顾

| 项 | 值 | 来源 |
|---|---|---|
| 机器人 | RM75-6F on Y-axis rail | `assets/robots/rm75_6f_8dof/RM75-6F-8dof.urdf` |
| DOF | 8 (rail_y ∈ [-0.18, +0.18] m + joint_1..7) | 同上 |
| 建图时基座 | 固定（rail_y=0，只用 7-DOF 臂） | 用户指定 |
| 体素 (位置) | 30 mm 立方体 | 用户指定 |
| 方向 (5-DOF) | ≈15° 分辨率 | 用户指定 |
| Roll (可选 6-DOF) | 15° | 扫描允许时 roll 可省略 |
| TCP 帧 | `tcp`（`link_7` 沿 +Z 220 mm） | URDF `link_7_to_tcp` |
| 底座候选变量 | **仅** rail 底座世界 y_b（1-D） | 物理只有 Y 向；x_b, z_b 由机架固定 |
| 参考论文 A | Zacharias et al. 2013 – Capability Map | PDF |
| 参考论文 B | Vahrenkamp et al. 2013 – Reachability Inversion | PDF |
| 已有 IK 环 | `pose_ik.solve_pose_ik` (WBC QP, 8-DOF) | [rm75_control/control/joint_admittance_8dof/pose_ik.py](rm75_control/control/joint_admittance_8dof/pose_ik.py) |
| 已有 FK | `RobotKinematics` (Pinocchio) | [rm75_control/control/joint_admittance_8dof/model.py](rm75_control/control/joint_admittance_8dof/model.py) |

**关键结论**：因为建图时 rail_y=0，图坐标系就是 `rail_base` = `slider_link` = `base_link` 帧
（`arm_mount` 固定 0 位移）。在线阶段任意 rail_y 只是把整张图沿 +Y 平移 rail_y —— 这正是
Vahrenkamp-2013 inversion 的核心平移不变性。

---

## 1. 目录结构 (在 `rm75_control/tools/` 下新建)

```
rm75_control/tools/
├── __init__.py                              # (已存在，追加 export)
├── state_echo.py                            # (已存在)
└── reachability/                            # ← 本次新增的顶层子包
    ├── __init__.py
    ├── README.md                            # 用法 & 关键概念速查
    │
    ├── data_model/                          # 无副作用的数据类 & IO 格式
    │   ├── __init__.py
    │   ├── voxel_grid.py                    # 3-D 位置离散化
    │   ├── orientation_grid.py              # 5-DOF 工具轴离散化 + 可选 roll
    │   ├── capability_map.py                # in-memory 结构 + save/load (npz+yaml)
    │   ├── schema.py                        # dataclasses & 版本号 (SCHEMA_VERSION="1.0")
    │   └── frames.py                        # rail_base ↔ world 帧变换、pose 类型别名
    │
    ├── kinematics/                          # 建图专用 IK / FK (轻量、可并行)
    │   ├── __init__.py
    │   ├── fk_batch.py                      # 批量 FK (numpy)，用于 Monte-Carlo 阶段
    │   ├── ik_dls.py                        # 阻尼最小二乘 IK (7-DOF, rail 锁 0)
    │   ├── ik_seeds.py                      # 多姿态种子池 (nominal + 镜像 + random)
    │   └── model_locked_rail.py             # 从 8-DOF URDF 派生 rail=0 的 Pinocchio model
    │
    ├── build/                               # 离线建图
    │   ├── __init__.py
    │   ├── config.py                        # BuildConfig + YAML loader
    │   ├── builder.py                       # 主流程（编排 MC + 可选 IK 精修）
    │   ├── mc_sampler.py                    # 正向 Monte-Carlo 采样填 bitmask
    │   ├── ik_refiner.py                    # 对边界体素做逆向 IK 补齐
    │   ├── workers.py                       # multiprocessing 工作池 (fork-safe)
    │   ├── metrics.py                       # D(x)、manipulability、条件数
    │   └── cli.py                           # python -m ...reachability.build
    │
    ├── inversion/                           # 在线查询 / 底座优化
    │   ├── __init__.py
    │   ├── loader.py                        # 快速 mmap 加载 capability_map
    │   ├── reach_set.py                     # 单 waypoint → 允许的 rail_y 集合 (1D 区间)
    │   ├── trajectory.py                    # Waypoint & ScanTrajectory dataclass
    │   ├── base_optimizer.py                # 完整扫描的最佳 y_b (Task-1)
    │   ├── prefix_solver.py                 # 前缀覆盖 (Task-2, 类 lazy-ORM 卷积)
    │   ├── quality.py                       # manipulability 加权、居中偏好
    │   └── cli.py                           # python -m ...reachability.inversion
    │
    ├── viz/                                 # 可视化 (Zacharias 风格球球图)
    │   ├── __init__.py
    │   ├── colormap.py                      # 与论文相同的 D(x) 配色 (blue→red)
    │   ├── sphere_glyphs.py                 # per-voxel 单色球体（快速全局图）
    │   ├── orientation_glyph.py             # per-voxel 方向 icosphere（细节图）
    │   ├── robot_scene.py                   # 加载 DAE 拼装机器人静态姿态
    │   ├── trajectory_overlay.py            # 画扫描轨迹 + 可达/不可达分段
    │   ├── inversion_scene.py               # 在物体/工件坐标系画 inverse map
    │   └── cli.py                           # python -m ...reachability.viz
    │
    ├── data/                                # gitignore；仅放小示例，正式数据放 /data/
    │   └── .gitkeep
    │
    └── tests/                               # 单元/集成测试
        ├── __init__.py
        ├── test_voxel_grid.py
        ├── test_orientation_grid.py
        ├── test_frames_shift_invariance.py
        ├── test_ik_dls_reachable_pose.py
        ├── test_mc_sampler_small.py
        ├── test_capability_map_io.py
        ├── test_reach_set_intersection.py
        ├── test_prefix_solver.py
        └── data/tiny_map_2cm_45deg.npz     # 迷你 map fixture
```

正式产出的地图数据放到仓库外或 `data/` 下（`.gitignore`）：

```
data/reachability/rm75_6f_3cm_15deg/
├── manifest.yaml            # 全部构建参数 + hash + git sha
├── voxels.npz               # 稀疏 (i,j,k) + D(x) + 若干统计
├── orientations.npy         # (N_ori, 3) 单位向量或 (N_ori, 4) quat
└── bitmask.npy              # (N_reachable_voxel, ceil(N_ori/8)) uint8
```

---

## 2. 关键数据结构 (`data_model/`)

### 2.1 [voxel_grid.py](rm75_control/tools/reachability/data_model/voxel_grid.py)

```python
@dataclass(frozen=True)
class VoxelGrid:
    origin_m: np.ndarray   # (3,) 世界/base 帧原点
    step_m: float = 0.03
    shape: tuple[int, int, int]  # (nx, ny, nz)

    def idx_of(self, p: np.ndarray) -> tuple[int,int,int]: ...
    def center_of(self, ijk) -> np.ndarray: ...
    def flat(self, ijk) -> int: ...
    def all_centers(self) -> np.ndarray:  # (N,3)
    def in_bounds(self, ijk) -> bool: ...
```

工作区默认 bounding box：x,y ∈ [-1.10, 1.10]，z ∈ [-0.30, 1.40]（RM75-6F 臂展 ≈ 1.09 m）。
默认 shape = (74, 74, 57) → ≈ 312 k 体素上限。**只对臂展球内**的体素做 IK/MC，
实际非空体素 ≈ 90 k。

### 2.2 [orientation_grid.py](rm75_control/tools/reachability/data_model/orientation_grid.py)

```python
class ToolAxisGrid:
    """5-DOF: 单位向量集合。默认 icosphere subdiv=3 → 642 点 ≈11°；
    可选 Fibonacci(N=200) ≈14°；用户 15° 请求映射到 subdiv=3。"""
    vectors: np.ndarray          # (N_ori, 3)
    def nearest(self, d: np.ndarray) -> int: ...   # KDTree
    def neighbors(self, i: int, half_angle_deg: float) -> list[int]: ...

class RollGrid:
    step_deg: float = 15.0
    angles: np.ndarray           # (24,)
```

**默认关闭 roll**（Zacharias 5-DOF）。当用户任务需要指定 roll 时打开 6-DOF 模式，
存储位数 24× 增长。

### 2.3 [capability_map.py](rm75_control/tools/reachability/data_model/capability_map.py)

```python
@dataclass
class CapabilityMap:
    grid: VoxelGrid
    orientations: ToolAxisGrid
    roll: RollGrid | None
    voxel_ids: np.ndarray            # (M,3) int32  只存非空体素
    bitmask: np.ndarray              # (M, ceil(N_ori/8)) uint8  或 (M, N_ori, ceil(N_roll/8))
    d_value: np.ndarray              # (M,) float32   ∈[0,1]
    mu_mean: np.ndarray | None       # (M,) 若做了 IK 精修则存
    meta: dict                       # {urdf_hash, git_sha, build_wall_s, ...}

    def query(self, ijk, tool_axis_idx) -> bool: ...
    def d_grid(self) -> np.ndarray:  # (nx,ny,nz)，未知处 NaN
    def save(self, dir_path: Path) -> None:
    @classmethod
    def load(cls, dir_path: Path, mmap: bool = True) -> "CapabilityMap":
```

**存储格式**：`voxels.npz` (稀疏 index+d)、`orientations.npy`、`bitmask.npy`；用 numpy
`memmap` 加载以便 100 MB 级 map 常驻。所有 float32 落盘。

### 2.4 [frames.py](rm75_control/tools/reachability/data_model/frames.py)

- `pose6` 别名 = ndarray (x,y,z,rx,ry,rz)
- `pose7` = (x,y,z,qx,qy,qz,qw)
- 关键工具：`shift_point_to_base(p_world, rail_base_world_xyz, rail_y) -> p_base`
- 单元测试专属：`test_frames_shift_invariance.py` 确认「map 平移不变性」——
  在 rail_y=0 建的 map，通过 (p_world − (0, y_b + rail_y, 0)) 转到 base 帧，
  查表结果 ≡ 直接 8-DOF FK/IK 在同 pose 上的结果。

---

## 3. 建图 IK / FK 后端 (`kinematics/`)

### 3.1 [model_locked_rail.py](rm75_control/tools/reachability/kinematics/model_locked_rail.py)

用 Pinocchio 的 `buildReducedModel(...)` 从 8-DOF URDF 冻结 `rail_y=0`，得到 7-DOF 模型。
好处：所有下游批量算子的自由度只有 7，SVD/Jacobian 更小；同时 URDF 只维护一份。

```python
def build_locked_rail_model(urdf_path=DEFAULT_URDF, rail_pos=0.0) -> pin.Model
def build_kin(model) -> pin.Data
```

### 3.2 [fk_batch.py](rm75_control/tools/reachability/kinematics/fk_batch.py)

`fk_positions_orientations(model, data, Q_arm: (K,7)) -> (positions:(K,3), quats:(K,4))`。
串行调用 `pin.forwardKinematics + updateFramePlacement(tcp)`；单线程 ≈ 4-6 万次/秒。
用于 Monte-Carlo 阶段。

### 3.3 [ik_dls.py](rm75_control/tools/reachability/kinematics/ik_dls.py)

**为什么不复用 `pose_ik.solve_pose_ik`**：QP 单次 ≈ 5-15 ms、循环 500 iter，太慢；
建图/精修阶段每帧只要「能不能达到」，不需要 CBF/nullspace/rail 处理。

```python
def ik_dls(model, data, tcp_id,
           pose_target: pose7,
           q_seed: np.ndarray,          # (7,)
           *,
           max_iter: int = 40,
           tol_pos_m: float = 5e-4,
           tol_rot_rad: float = 5e-3,
           lam: float = 0.05,           # 阻尼系数
           limits_lo: np.ndarray,       # (7,) rad
           limits_hi: np.ndarray,
           ) -> tuple[np.ndarray, bool, dict]:
    """标准 SR-inverse：dq = J^T (J J^T + λ^2 I)^-1 e，e 使用 log6 姿态误差。
    末尾把 q 夹到 limits ± 1e-3；若夹后 pose 误差重新超过 tol → 返回 ok=False。
    单次目标 ≤ 200 μs（无 QP，无 constraint）。"""
```

### 3.4 [ik_seeds.py](rm75_control/tools/reachability/kinematics/ik_seeds.py)

对 7-DOF 冗余臂，一次种子往往落到不合的分支。构造多种子池：

- `nominal`：即 `joint_admittance_8dof` YAML 中 `q_nominal_deg`（右上 45°/肘弯 90° 姿）
- `mirror_elbow_up/down`
- `pseudo-random`：Halton 序列，8 或 16 个
- `analytical_seeds`：给定 wrist 位置 → 用 3R 位置 IK（joint_1..joint_3）+ 球腕 3-1-3 欧拉解析式生成 ≤ 8 组解 (可选，V2)

选择策略：`ik_dls` 从 `seeds` 顺序尝试直到 ok，或全部尝试并保留 manipulability 最大者。

---

## 4. 离线建图 (`build/`)

### 4.1 [config.py](rm75_control/tools/reachability/build/config.py)

```python
@dataclass
class BuildConfig:
    urdf_path: Path = DEFAULT_URDF
    tcp_frame: str = "tcp"
    grid: VoxelGridConfig    # origin, shape, step (30 mm)
    orient: OrientationGridConfig  # subdiv=3 (or fibonacci N=200); with_roll=False
    mode: Literal["mc", "ik", "hybrid"] = "hybrid"
    mc: MonteCarloConfig     # n_samples=1e7, seed=0, batch=1_000_000
    ik: IkRefineConfig       # boundary_d_min=0.05, boundary_d_max=0.98,
                             # per_voxel_orient_budget=all_unreached
    workers: int = os.cpu_count()
    output_dir: Path
    save_manipulability: bool = True

def load_yaml(path: Path) -> BuildConfig: ...
```

CLI 默认 YAML：`configs/reachability/rm75_6f_3cm_15deg.yaml`（新增）。

### 4.2 [mc_sampler.py](rm75_control/tools/reachability/build/mc_sampler.py) — Phase 1

**思想**：Zacharias 论文中的 dense IK 太贵；先用正向 Monte-Carlo 得到 D(x) 下界，
再用 IK 精修边界（论文中 Fig. 6 附近的 tail-off 区域）。

```python
def run(model, data, tcp_id, cfg: BuildConfig) -> _MCAccumulator:
    lo, hi = model.lowerPositionLimit, model.upperPositionLimit  # 7-DOF
    for batch in split(cfg.mc.n_samples, cfg.mc.batch):
        Q = rng.uniform(lo, hi, size=(batch, 7))
        pos, quat = fk_batch(model, data, tcp_id, Q)
        ijk = grid.idx_of(pos)                 # (batch, 3)
        in_bb = grid.in_bounds(ijk)            # bool mask
        # 工具轴 = 姿态 R @ [0,0,1]（TCP z 轴，扫描/绘画常用）
        tool_axis = quats_z_axis(quat)         # (batch, 3)
        ori_idx = orient_grid.nearest_batch(tool_axis)  # KDTree
        accum.mark(ijk[in_bb], ori_idx[in_bb])          # 位与
        # 若开启 6-DOF：还要绕轴角度 → RollGrid.nearest → 另一 bitmask
    return accum
```

- `n_samples = 1e7`：单线程 ≈ 3-5 分钟；多进程按 `workers` 分片求和 bitmask 后 OR。
- 得到的 D(x) = 已覆盖方向数 / N_ori，是 **下界**（MC 覆盖率不完全）。

### 4.3 [ik_refiner.py](rm75_control/tools/reachability/build/ik_refiner.py) — Phase 2（可选）

针对 **边界体素**（0.02 < D(x) < 0.98）与它们的未覆盖方向：

```python
for v in boundary_voxels:
    for ori_idx in unreached_orientations(v):
        pose = build_pose(voxel_center(v), orient.vectors[ori_idx],
                          roll=0.0)  # 5-DOF：roll 任意，选 0
        q, ok, _ = ik_dls(model, data, tcp_id, pose,
                          seed_pool=seeds, ...)
        if ok:
            bitmask[v, ori_idx] = 1
            if cfg.save_manipulability:
                mu[v, ori_idx] = manipulability(J)
```

预算控制：`ik.per_voxel_budget_s = 0.2s`；超时跳过。多进程按体素分片。

### 4.4 [builder.py](rm75_control/tools/reachability/build/builder.py)

```python
def build(cfg: BuildConfig) -> CapabilityMap:
    model = build_locked_rail_model(cfg.urdf_path, rail_pos=0.0)
    grid  = VoxelGrid.from_config(cfg.grid)
    orient = ToolAxisGrid.from_config(cfg.orient)
    accum = MCAccumulator(grid, orient)
    mc_sampler.run(model, cfg, accum, workers=cfg.workers)
    if cfg.mode in ("ik", "hybrid"):
        ik_refiner.run(model, cfg, accum, workers=cfg.workers)
    cm = accum.freeze()                # 生成 CapabilityMap dataclass
    cm.meta.update({
        "urdf_sha256": sha256(cfg.urdf_path),
        "git_sha": git_head_sha(),
        "wall_s": ...,
        "mc_samples": cfg.mc.n_samples,
        "ik_refined": cfg.mode in ("ik","hybrid"),
    })
    cm.save(cfg.output_dir)
    return cm
```

### 4.5 [cli.py](rm75_control/tools/reachability/build/cli.py)

```
python -m rm75_control.tools.reachability.build.cli \
  --config configs/reachability/rm75_6f_3cm_15deg.yaml \
  --output data/reachability/rm75_6f_3cm_15deg \
  --workers 8 --mode hybrid
```

日志：进度条 (tqdm)、每 batch 一条 stderr、结束打印 `D(x)` 直方图 + 覆盖体素数。

**预算估计**（Ryzen 8 核）：
- MC 1e7 样本 ≈ 3 min（单核 4-6 万 FK/s，8 核 30-40 万/s）。
- IK 精修（默认 ≈ 30 k 边界体素 × 平均 20 未覆盖方向 × 200 μs × 4 seed）≈ 25 min。
- 磁盘 ≈ 50-80 MB。

---

## 5. 在线查询 & 底座优化 (`inversion/`)

### 5.1 [trajectory.py](rm75_control/tools/reachability/inversion/trajectory.py)

```python
@dataclass
class Waypoint:
    p_world: np.ndarray                 # (3,)
    tool_axis_world: np.ndarray         # (3,) 单位向量，扫描表面外法向 (通常 = TCP z 轴指向工件)
    axis_tol_deg: float = 10.0          # 允许偏离多少度（→ 邻居方向索引）
    pos_tol_m: float = 0.015            # 允许位置偏差（→ 邻居体素）
    roll_range_deg: tuple[float,float] | None = None
    weight: float = 1.0                 # 打分权重（前缀阶段等间距 1.0）

@dataclass
class ScanTrajectory:
    waypoints: list[Waypoint]
    # 世界系；建议由用户从 CAD/示教得到
    def arc_length_m(self, i: int) -> float
```

### 5.2 [reach_set.py](rm75_control/tools/reachability/inversion/reach_set.py) — 核心变换

给定固定 `rail_base_world = (x_b, y_b, z_b)`：

```python
def allowed_y_shift(cm: CapabilityMap,
                    wp: Waypoint,
                    rail_base_xyz_world: np.ndarray,
                    *,
                    y_shift_range: tuple[float,float],
                    y_shift_step: float | None = None,
                    ) -> IntervalSet:
    """返回让 waypoint 可达的所有 y_shift = y_b + rail_y 的一维区间集合。
    y_shift ∈ [-∞, +∞]，最后再和 [y_b - 0.18, y_b + 0.18] 求交。"""
    # 1) 目标在 base 帧的位置：p_base = p_world - rail_base_xyz_world - (0, y_shift, 0)
    # 2) 因为要求 arm-base 帧 y 分量：p_b_y = p_wy - z_b_y - y_shift
    # 3) 遍历 y_shift 采样（步长 = grid.step_m/2 = 15 mm）：
    #    - 找到体素 ijk
    #    - 查邻居方向集合（对应 wp.axis_tol_deg，KDTree 内部预算）
    #    - 若 bitmask 有任一命中 → 该 y_shift 可行
    # 4) 用 run-length 压缩得到区间集合 (IntervalSet)
```

`IntervalSet` 是简单闭区间列表（合并、交、总长度、包含性检查）。测试见
`test_reach_set_intersection.py`。

**加速**：
- 因为方向邻居集合仅由 (wp.tool_axis_world, axis_tol_deg) 决定，同一 wp 里预计算一次。
- y_shift 采样步 = 15 mm，一 waypoint ≤ ~24 次查表（0.36 m / 15 mm），单条查表微秒级。

### 5.3 [base_optimizer.py](rm75_control/tools/reachability/inversion/base_optimizer.py) — Task 1

「完整扫描」求最佳 y_b：

```python
def full_scan_best_yb(cm, traj, *,
                      xz_base_world: tuple[float,float],
                      rail_travel_half: float = 0.18,
                      yb_range: tuple[float,float] = (-1.0, 1.0),
                      yb_step: float = 0.01,
                      quality: QualityWeights,
                     ) -> FullScanResult:
    # 1) 对每个候选 y_b：
    #    for i, wp in enumerate(traj):
    #        S_i = allowed_y_shift(cm, wp, (x_b, y_b, z_b))
    #        A_i = S_i ∩ [y_b - 0.18, y_b + 0.18]
    #        if A_i is empty: 该 y_b 失败, break
    # 2) 成功者按 quality 打分，取最大：
    #    score = Σ_i w_i * mu_hat(wp_i, chosen_rail_y_i)
    #          + λ_center * (-mean rail_y^2)
    #          + λ_uniform * (-var of rail_y)
    # 3) 也给出各 waypoint 对应的 rail_y_i（选 A_i 内 argmax mu；插值平滑）
```

返回 `FullScanResult(y_b_best, rail_y_series, score, feasible_yb_intervals)`。

**明确决定：只搜 y_b (1-D)。** 函数签名接受 `xz_base_world` 只作为常量输入（机架实测值），
不做 x_b 扫描；如未来真的需要，另开分支实现，不进本次范围。

### 5.4 [prefix_solver.py](rm75_control/tools/reachability/inversion/prefix_solver.py) — Task 2

「不能整段则求最长前缀」，形式化：

给定固定 y_b（外循环枚举），前缀可行 ⟺ 累计交 `⋂_{j≤i} A_j ≠ ∅`。

```python
def longest_prefix(cm, traj, *,
                   xz_base_world,
                   yb_range, yb_step, rail_travel_half=0.18,
                   ) -> PrefixResult:
    # 关键观察：A_j 只与 (y_b, wp_j) 相关；对每个 y_b 做一次 forward sweep。
    #   inter = IntervalSet.full()
    #   for j, wp in enumerate(traj):
    #       inter = inter ∩ A_j(y_b, wp)
    #       if inter.empty:
    #           record last_reached = j - 1, best_endpoint_arc_len; break
    # 输出：所有 y_b 上的 (last_reached_index, last_reached_arc_len)
    # 选择准则：优先 last_reached_index 大；同 index 下选 arc_len 更大的
    #           (处理 waypoint 间隔不均的情况)；同分再看 quality。
```

若还要求「稍稍越过失败点也算」（用户所说“类似拿 lazy ORM 做卷积”容错）：
- 对失败 waypoint j：把 wp_j 的 `axis_tol_deg` 与 `pos_tol_m` 乘 2 再试一次；
  若仍失败则 j 定为终止点（并在报告里给「置信度 = strict / relaxed」两档）。

**「lazy-ORM 卷积」实现**：把每个 waypoint 的 A_j 视作 y_b 轴上的 0/1 指示函数
`f_j(y_b) = 1[A_j ≠ ∅]`；累积交 = `min_j f_j`（滑动 AND）。
物理上就是「随着 j 递增，可用 y_b 集合单调收缩」——
可以按 y_b 列出「首次归零时刻」的映射 `t*(y_b)`，t\*(y_b) 就是该 y_b 下能扫完的最长前缀 waypoint 数。
最终选 `argmax_{y_b} t*(y_b)`（同分选 quality）。整个 sweep 复杂度 O(N_wp · N_yb · N_yshift)。

### 5.5 [quality.py](rm75_control/tools/reachability/inversion/quality.py)

```python
@dataclass
class QualityWeights:
    manipulability: float = 1.0
    center_rail: float = 0.2   # 惩罚 rail_y 靠近极限
    smooth_rail: float = 0.5   # 惩罚 rail_y 相邻 waypoint 抖动
    d_neighbor: float = 0.1    # 惩罚 D(x) 边缘
```

分数用 waypoint 处从 `cm.mu_mean` 查得（若 map 有）；否则用 `1 - dist_to_rail_limit`。

### 5.6 [cli.py](rm75_control/tools/reachability/inversion/cli.py)

```
python -m rm75_control.tools.reachability.inversion.cli \
  --map data/reachability/rm75_6f_3cm_15deg \
  --trajectory examples/scan_paths/board_left_to_right.json \
  --xb 0.0 --zb 0.0 \
  --yb-range -0.5 0.5 --yb-step 0.01 \
  --mode full           # or "prefix" / "both"
  --report /tmp/base_report.json --plot /tmp/base_plot.png
```

**输出报告 (JSON)**：
```json
{ "mode": "both",
  "full_scan": { "feasible": true, "y_b_best": 0.043, "score": 12.7,
                 "rail_y_series": [ ... ] },
  "prefix":    { "y_b_best": 0.021, "last_wp_index": 217, "arc_len_m": 1.34,
                 "rail_y_series": [...], "relaxed": false },
  "meta": { "map_dir": "...", "traj_hash": "...", "elapsed_s": 4.2 } }
```

---

## 6. 可视化 (`viz/`) —— 严格对齐 Zacharias 2013 + Vahrenkamp 2013

所有出图脚本按论文图逐图对标；每种视图对应一个函数、一个 CLI 子命令、一个 golden
截图 fixture（用 `pytest-image-diff` / SSIM 阈值卡回归）。默认输出白底、无坐标轴、
等距正交相机、`figsize=(1600,1200)` PNG + 可选 `.svg` 矢量版。

### 6.1 依赖

- `pyvista>=0.43`（VTK 后端，出等距投影正交图；PDF/SVG 用 `vtk.vtkGL2PSExporter`）
- `trimesh>=4.0` + `pycollada>=0.7`（读 `.dae`；`pv.wrap(trimesh_mesh)` 转 PolyData）
- `matplotlib>=3.8`（配色/直方图/2-D 平面切片）
- `imageio>=2.30`（PNG 输出、可选 GIF 旋转视频）

### 6.2 论文图 ↔ 本项目脚本 对照表

| 论文原图 | 本项目文件 / 函数 | CLI 子命令 |
|---|---|---|
| **Zacharias Fig 3** —— 体素中心球，色 = D(x)，机器人在中心 (rest pose) | `sphere_glyphs.py::render_reachability_index` | `viz capability` |
| **Zacharias Fig 4/5** —— 每个体素一个 **reachability sphere**，球面上小 patch 颜色 = 该方向是否可达 | `orientation_glyph.py::render_direction_spheres` | `viz directions` |
| **Zacharias Fig 6** —— 用 **shape primitive**（cone / disk / cylinder）拟合方向集合，抽象化每个体素的可达锥 | `orientation_glyph.py::render_shape_primitives` (V2, 见 §6.8) | `viz primitives` |
| **Zacharias Fig 8/9** —— 通过水平面 (z = const) / 竖直面 (y = 0) 的切片图 | `sphere_glyphs.py::render_slice` | `viz slice --plane z=0.5` |
| **Vahrenkamp Fig 3** —— 工件坐标系内的 **Inverse Reachability Map**，底盘 (x,y) 上颜色 = 可行性密度 | `inversion_scene.py::render_irm_ground` | `viz irm` |
| **Vahrenkamp Fig 4** —— 目标物体旁若干候选 base 位置，颜色 = score | `inversion_scene.py::render_base_candidates` | `viz placement` |
| **Vahrenkamp Fig 6** —— 选定最佳 base 时的机器人抓取姿态叠加 | `inversion_scene.py::render_best_placement_pose` | `viz placement --best-pose` |

### 6.3 [colormap.py](rm75_control/tools/reachability/viz/colormap.py) — 严格论文配色

- **Zacharias colormap** (`zacharias_d`):
  Zacharias Fig 3 用的是从深蓝 (D≈0) → 青 → 绿 → 黄 → 红 (D≈1) 的类 `jet` 段，
  我们用 matplotlib `"turbo"` 的截取 [0.05, 0.95]（视觉最接近，避免 jet 的暗端伪影）；
  再暴露 `zacharias_d_paper = LinearSegmentedColormap(...)` 用 5 段手写色标以做最严格复刻。
- **Zacharias direction colormap** (`zacharias_dir`):
  Fig 4/5 里 reachable direction = 绿色 `#3faa3f`，unreachable = 灰色 `#c8c8c8`，
  面片间黑色 0.3 pt 描边。
- **Vahrenkamp IRM colormap** (`vahrenkamp_irm`):
  论文 Fig 3 是 blue→cyan→green→yellow→red 的密度 colormap（"viridis" 反向或
  自定义），score / count 归一化后映射。
- **Vahrenkamp placement score colormap** (`vahrenkamp_score`):
  Fig 4 的球颜色：低分红、高分绿 —— 直接 `RdYlGn`（matplotlib 自带），
  最优点用金色球 (`#ffd700`) 高亮。

所有 colormap 都提供 `mpl_cmap()` 与 `pv_lookup_table()` 两套接口。

### 6.4 [robot_scene.py](rm75_control/tools/reachability/viz/robot_scene.py)

```python
def build_robot_pv(urdf_path: Path,
                   q_full: np.ndarray | None = None,
                   base_pose_world: SE3 = SE3.identity(),
                   *,
                   uniform_gray: bool = True,  # 论文里机器人常统一浅灰
                   ) -> pv.MultiBlock:
    """遍历 URDF 各 link <visual>，Pinocchio FK 得到每 link SE3，trimesh 载入 DAE
    → PolyData → transform；返回 MultiBlock，方便 pl.add_mesh(mb, color='lightgray')。
    默认 q_full = zeros（rail=0, joint_1..7=0），与两篇论文里的 rest pose 惯例一致。"""
```

- 首选 URDF：`assets/robots/rm75_6f_8dof/RM75-6F-8dof.genesis.urdf`（含 visual 段）。
- `uniform_gray=True` 时忽略 DAE 材质，用 `#b5b5b5` + 环境光；这是 Zacharias Fig 3 的机器人配色。
- 提供 `add_rest_pose_annotation(pl, robot_mb)`：论文里常在机器人脚下加"世界坐标系
  三色轴 (X 红 / Y 绿 / Z 蓝)"和一个浅色地面 disk r=1.2 m —— 复刻这一点。

### 6.5 [sphere_glyphs.py](rm75_control/tools/reachability/viz/sphere_glyphs.py)  → Zacharias Fig 3

```python
def render_reachability_index(cm: CapabilityMap,
                              out_png: Path,
                              *,
                              robot_urdf: Path | None = None,
                              d_min: float = 0.02,
                              sphere_radius_m: float = 0.010,   # 体素 1/3
                              cmap: str = "zacharias_d",
                              camera: str = "iso_zacharias",     # 见下
                              show_axes: bool = False,
                              show_colorbar: bool = True,
                              show_ground_disk: bool = True,
                              screenshot_size=(1600, 1200)):
```

- 相机预设 `iso_zacharias`：`camera_position=(2.2, -2.2, 1.6)`, `focal_point=(0,0,0.6)`,
  `viewup=(0,0,1)`, `parallel_projection=True`（论文用正交投影而非透视）。
- 颜色条水平放在图下方 (`pl.add_scalar_bar(vertical=False, position_x=0.25, position_y=0.05, width=0.5)`)，
  标签 "Reachability index D(x)"（与论文一致）。
- 背景纯白 `pl.background_color='white'`；灯光用 `pl.enable_lightkit()` 的三点布光。
- 输出 `.png` + 可选 `.svg`（`pl.save_graphic(...)`）。

`render_slice(cm, plane_spec="z=0.5", thickness_m=0.03, ...)` → Zacharias Fig 8/9：
- 只保留该平面 ±厚度内的体素；相机自动切换到俯视 / 侧视正交。
- 论文里切片图常配一根"当前切片高度"标注在小 3-D 缩略图上 —— 用 `pv.Chart2D` 加子窗
  展示。

### 6.6 [orientation_glyph.py](rm75_control/tools/reachability/viz/orientation_glyph.py) → Zacharias Fig 4/5

**这是 Zacharias 论文最标志性的图**：每个体素画一个大球，球面按方向 icosphere 三角剖分，
每个 face 颜色 = 该 face 中心方向是否可达（绿/灰）。

```python
def render_direction_spheres(cm: CapabilityMap,
                             out_png: Path,
                             *,
                             voxel_selector: Callable[[VoxelId], bool] | None = None,
                             stride: int = 4,                # 每 stride 体素画一个（避免过密）
                             sphere_radius_m: float = 0.012,
                             face_reachable_color: str = "#3faa3f",
                             face_missing_color: str = "#c8c8c8",
                             edge_color: str = "#000000", edge_width: float = 0.3,
                             robot_urdf: Path | None = None,
                             ):
```

- 内部：`orientations.vectors` 已是 icosphere subdiv=3 的 642 顶点；把它转成 pv.PolyData
  三角面片；per-face `active_scalar` = bitmask 命中 → 二值 colormap。
- 顶点法向 = 顶点位置（单位球），保证光照一致。
- 与 §6.5 可组合：先出全局 D(x) 球图选感兴趣区域，再在该区域跑本函数看方向分布。

**交互变体** `render_direction_spheres_interactive(cm, ...)`：
- `pl.enable_point_picking(callback=..., use_mesh=True)`：点击一个 sphere_glyph 弹出该体素
  的方向球细节窗，符合 Zacharias 论文 §V-B 描述的 "inspect a voxel" 工作流。

### 6.7 [trajectory_overlay.py](rm75_control/tools/reachability/viz/trajectory_overlay.py)

自我扩展（论文里没有对应图，但用来展示扫描 + 前缀结果）：
- 轨迹折线：可达段 `#3faa3f`、不可达段 `#d94040`、宽度 4 px；
- 在可达段等距画 tool-axis 小箭头（与 Waypoint.tool_axis_world 一致）；
- 用金色球标注 `last_reached_waypoint`（对应 §5.4 prefix_solver 输出）。

### 6.8 [inversion_scene.py](rm75_control/tools/reachability/viz/inversion_scene.py) → Vahrenkamp Fig 3/4/6

严格对齐 Vahrenkamp 三张核心图：

**A. `render_irm_ground` → Vahrenkamp Fig 3**
把 IRM 投影到 base 底盘平面（对我们即是 y_b 一维；论文里是 x_b, y_b 二维平面）。
做法：
- 在工件（把用户第一 waypoint 当参考"target"）周围一个可视半径 (默认 1.5 m) 内
  沿 y_b 轴布 200 个候选球；
- 球色 = "该 y_b 下能覆盖的 waypoint 数量 / 总数"（Vahrenkamp 论文里是 density；
  用 `vahrenkamp_irm` colormap）；
- 由于我们是 1-D，把球压成"沿 y 方向一排"，视觉上是一条彩色项链——同时叠加一个
  水平柱状图 (`pv.Chart2D`) 展示 sweep 曲线，让 1-D 结果依然直观。
- 场景中心放 target 用一个 `pv.Sphere(0.03)` 高亮红色（论文里 target 就是这么标）。

**B. `render_base_candidates` → Vahrenkamp Fig 4**
把 `full_scan_best_yb` 或 `longest_prefix` 返回的 feasible y_b 集合画出：
- 每个可行 y_b 一个球；`vahrenkamp_score` colormap；
- 最优点金球 (`#ffd700`) + 简短文字 "y_b* = 0.043 m"（`pl.add_text`）；
- 灰色球代表不可行的 y_b（透明度 0.3）；论文 Fig 4 就是这个"可行/不可行 + 最优点"三态。

**C. `render_best_placement_pose` → Vahrenkamp Fig 6**
选定 best y_b 后，用 `pose_ik.solve_pose_ik` 反解某个代表 waypoint（默认轨迹中点）的
q_full，再用 `robot_scene.build_robot_pv` 把机器人渲染在场景里；工件同时显示。
论文 Fig 6 就是"最优 base + 抓取姿态"复合图，我们做的是"最优 y_b + 扫描代表姿态"。

### 6.9 [cli.py](rm75_control/tools/reachability/viz/cli.py)

```
# Zacharias 图
viz capability     --map ... [--slice z=0.5] --out cap.png
viz directions     --map ... --stride 4 --out dirs.png
viz slice          --map ... --plane z=0.5 --out slice_z050.png

# Vahrenkamp 图
viz irm            --map ... --trajectory scan.json --out irm.png
viz placement      --map ... --trajectory scan.json [--best-pose] --out placement.png
```

每条命令统一支持：`--white-bg / --parallel / --figsize 1600 1200 / --svg`
以保持论文 figure ready 的输出。

### 6.10 论文风格 QA / 回归

- `tests/data/golden/zacharias_fig3_like.png` 等 5 张 golden 图（分辨率 800×600 缩略）
  用 `pytest --run-viz` 手工跑 + `SSIM ≥ 0.95` 判定，避免脚本改动破坏视觉风格。
- 每个 viz 函数在 docstring 顶部注明 "对应论文图 X" + 主要视觉参数（相机、colormap、
  背景、光照），后续 review 时一眼可对照。

---

## 7. 测试策略 (`tests/`)

| 文件 | 检查点 |
|---|---|
| `test_voxel_grid.py` | idx/center 双向一致；越界返回；all_centers 形状 |
| `test_orientation_grid.py` | icosphere subdiv=3 → 642 vertex；nearest 精度 ≤ subdiv 步长 |
| `test_frames_shift_invariance.py` | 用 8-DOF FK 在 rail_y=r 得到 pose P；用 7-DOF FK 在 rail_y=0 得到 pose P'，比对 P' + (0,r,0) == P |
| `test_ik_dls_reachable_pose.py` | 对若干已知可达 pose，`ik_dls` 5×5 抽样成功率 ≥ 95% |
| `test_mc_sampler_small.py` | 小 grid (10 mm, 30°) + 1e5 样本，跑完 & bitmask non-zero；确定性 seed |
| `test_capability_map_io.py` | round-trip save/load bit-exact；mmap 打开可读 |
| `test_reach_set_intersection.py` | 手工构造 3 waypoint & 小 map fixture → 交集正确 |
| `test_prefix_solver.py` | 构造「前 5 可达、第 6 不可达」轨迹 → last_index==4；relaxed 模式 last_index==5 |
| `tests/data/tiny_map_2cm_45deg.npz` | ≤ 200 KB 的最小 fixture，CI 里 <10 s 跑完 |

`pytest -q rm75_control/tools/reachability/tests` 应无外部依赖（除 numpy / scipy / pinocchio）。
可视化测试用 `pyvista.OFF_SCREEN=True` + `pytest.mark.skipif(no display)`。

---

## 8. 与现有代码的连接点

- **不改**：`control/joint_admittance_8dof/*`（已经稳定，不引入循环依赖）。
- **复用**：
  - `RobotKinematics` 里的 `urdf_path` 常量与 `EXPECTED_NQ=8` 结构 → `kinematics/model_locked_rail.py` 直接读同一 URDF。
  - `pose_ik.solve_pose_ik`：**只在验证脚本**里用 (作为 gold IK 对比 `ik_dls`)。
  - `configs/joint_admittance_8dof.yaml` 中的 `q_nominal_deg`：`ik_seeds.py` 默认 seed。
- **新增**：`configs/reachability/rm75_6f_3cm_15deg.yaml`（BuildConfig 默认）。
- **`tools/__init__.py`**：追加子包 re-export，让 `python -m rm75_control.tools.reachability.build.cli` 可用。

依赖清单追加到根 `requirements.txt`：
```
pyvista>=0.43
trimesh>=4.0
pycollada>=0.7
imageio>=2.30
tqdm>=4.66
```
（scipy / numpy / pinocchio / yaml 仓库已具备。）

---

## 9. 开发顺序 (推荐 PR 拆分)

1. **PR1 — 骨架 + 数据模型**：目录、`data_model/*`、`kinematics/model_locked_rail.py`、
   `kinematics/fk_batch.py`、`tests/test_voxel_grid.py` / `test_orientation_grid.py` /
   `test_frames_shift_invariance.py`。
2. **PR2 — IK 后端**：`kinematics/ik_dls.py` + `ik_seeds.py` + 单测。
3. **PR3 — MC 建图**：`build/mc_sampler.py` + `builder.py`（只支持 mode="mc"）+ CLI +
   `configs/reachability/rm75_6f_3cm_15deg.yaml`。产出第一份 map。
4. **PR4 — 可视化基础 (Zacharias Fig 3)**：`viz/colormap.py` (zacharias_d + vahrenkamp_*) +
   `viz/robot_scene.py` + `viz/sphere_glyphs.py::render_reachability_index / render_slice` +
   CLI `viz capability / viz slice`；golden 截图 `zacharias_fig3_like.png` 通过 SSIM 回归。
5. **PR5 — IK 精修**：`build/ik_refiner.py`，mode="hybrid"；对比 MC 前后 D(x) 直方图。
6. **PR6 — 查询/优化**：`inversion/*` 全部 + 单测。
7. **PR7 — 论文级可视化**：
   - Zacharias Fig 4/5 方向球：`orientation_glyph.py::render_direction_spheres` + interactive picking
   - `trajectory_overlay.py`（可达/不可达分段 + tool-axis 箭头 + 前缀终点金球）
   - Vahrenkamp Fig 3/4/6：`inversion_scene.py::render_irm_ground / render_base_candidates / render_best_placement_pose`
   - 完整 viz CLI + 5 张 golden 图 + SSIM 回归。

每个 PR ≤ ~800 行改动，独立可跑。

---

## 10. 主要风险 & 缓解

| 风险 | 缓解 |
|---|---|
| IK 精修慢到不能忍 | 默认 mode="mc"，只在 hybrid 中启用；per-voxel wall budget；进度可 checkpoint 断点续跑 |
| 存储爆炸（6-DOF roll 打开） | 默认关 roll；BuildConfig 里显式 opt-in；bitmask 用位打包 |
| DAE 加载失败（老 trimesh 版本对某些 collada 支持差） | 提供 `--robot-urdf simple` fallback，画 base+link 胶囊体 |
| 8-DOF 与 7-DOF 语义混淆 | `frames.py` 里所有 shift 用 typed `RailBaseWorld`, `ArmBaseFrame` 名字包裹；`test_frames_shift_invariance.py` 卡住底线 |
| 轨迹方向惯例（TCP z 向工件 or 反向？） | `Waypoint.tool_axis_world` 明确注释「指向体素中心的外法向 = TCP +z 期望方向」；CLI 提供 `--flip-tool-axis` |
| 底座 x_b 是否也搜索 | **不搜**。物理只有 Y 向轨；x_b 由机架实测直接传入 |

---

## 11. 与两篇论文的对应关系

- **Zacharias 2013 Capability Map**：
  - 体素 + 方向离散化 → §2.1 / §2.2
  - D(x) 定义与球球图 → §6.3
  - Monte-Carlo/IK 双阶段（论文以 IK 为主，我们做混合以省时） → §4
- **Vahrenkamp 2013 Reachability Inversion**：
  - 把 map 反向到「工件坐标系」查询允许基座 → §5.2 的 `y_shift` 变换正是 1D 版
  - 「Base placement scoring」= §5.5 QualityWeights
  - 前缀覆盖为本项目扩展（论文只做静态点抓取，我们做扫描轨迹的滑动可达）→ §5.4

---

## 12. 快速验收清单 (Definition of Done)

- [ ] `python -m rm75_control.tools.reachability.build.cli` 在默认参数下 15 分钟内产出 map。
- [ ] `viz capability` 出图与 **Zacharias Fig 3** 视觉一致：正交等距相机、白底、`zacharias_d`
      colormap 深蓝→红、机器人统一浅灰、地面浅色 disk、水平 colorbar 标 "D(x)"；
      SSIM(golden) ≥ 0.95。
- [ ] `viz directions` 出图与 **Zacharias Fig 4/5** 视觉一致：per-voxel 方向球，
      reachable face `#3faa3f`、missing face `#c8c8c8`、黑色描边。
- [ ] `viz placement` 出图与 **Vahrenkamp Fig 4** 视觉一致：可行球用 `RdYlGn` 上色、
      最优球金色 `#ffd700` 高亮 + 文字标注；不可行球灰透明。
- [ ] 已知可达 pose（`joint_admittance_8dof` YAML 的 `q_nominal_deg` FK 结果）在 map 里 D(x) ≥ 0.4。
- [ ] 已知不可达 pose（如工作区外 2 m）在 map 里返回“未采样”。
- [ ] `inversion.cli --mode full` 对一条 30 waypoint 的示例扫描线，秒级返回 y_b_best，
      并且用 `pose_ik.solve_pose_ik` 在 8-DOF 上验证每 waypoint 都收敛。
- [ ] `inversion.cli --mode prefix` 对一条故意超长（>1 m）扫描线返回合理 last_wp_index，
      relaxed 与 strict 两档一致性正常。
- [ ] 全部单元测试通过；CI ≤ 60 s。
