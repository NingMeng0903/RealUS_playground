# 腿部拓扑修复与原生 MedSeg 标注验证

在 MedSeg 原生 CT 中，右股总静脉（23）和右股静脉（39）从第 0 层到第 30 层连续可见。相邻层中心间距 3 mm，首末层中心相距 90 mm。两标签在第 12 层和第 13 层之间共享 233 个体素面。目标路径有 31 个源标注节点、30 条体素接触边、1 个连通分量，插值节点为 0。

同一例 CT 在中立位 SMPL-X 中从 −561.19 到 −477.81 mm。原扫查程序使用独立的 5 mm 对齐网格，目标在 −480 至 −560 mm 的 17 个节点也全部连通，−525 到 −530 mm 的链接经过标签交界。七组形体共用该病例的同一套源标签。原生 CT 证据仅覆盖近端大腿。

## 图谱和留出解剖

| 来源 | 节点 | 边 | 分量 | 推断边 |
| --- | ---: | ---: | ---: | ---: |
| BodyParts3D 右腿 | 3,847 | 3,827 | 20 | 3 |
| BodyParts3D 左腿镜像 | 3,703 | 3,680 | 23 | 1 |
| Open3DModel 右腿 | 4,100 | 4,068 | 32 | 12 |
| 已配准 MedSeg | 330 | 322 | 8 | 0 |
| Blender 右腿留出 | 2,392 | 2,384 | 8 | 0 |

- 每条具名血管独立切片，保留同层多个环。同一源网格分量内使用双向最近与竞争距离检查关联相邻环。候选短断点还要通过两端局部方向、稠密原曲面复切、竞争环、体表与骨占据率检查。
- Open3DModel 右大隐静脉在 741 到 749 mm 之间有 8 mm 的轴向断点。0.5 mm 密度复切取得 15/15 个源曲面闭环，加入 7 个推断节点和 8 条推断边。其他图谱推断边是同源、已声明的主干接合，端部临近源曲面的证据随边保存。
- 左镜像大隐静脉 57 mm 缺口、Open3DModel 大隐静脉 124 mm 缺口与腓静脉 17 mm 缺口保留断开。主干转接和侧支身份单独记录。原生 CT 的 103 个被拒接触包含 73 个跨动静脉接触和 30 个未声明的不同身份接触。
- Blender 留出中，三处主静脉截段共用源中心线端点，大隐静脉汇入点也为同源精确命中。动脉三个端点间距约 18.82、34.25 和 57.92 mm，维持断开。Blender 几何和链接只供留出评估。
- 15 mm 用于列出长缺口，12 mm 用于限制推断的最大轴向端点距离。方向、局部接触和占据率阈值是这批几何的保守工程参数。图谱连续性和超声可见性应各自评估。

## 文件和接口

| 文件 | 用途 |
| --- | --- |
| medseg_ct_route.npz | 原生 31 层的 RAS 位置、标签编号、相邻层共享面数和七组 SMPL-X 形体坐标 |
| medseg_native_graph.json | CT 源节点、逐体素接触边、已证实接点和被拒接触 |
| ct_geometry.npz | 原连续扫查格式 ys、origins、radii、centers、status、ct_bounds、neutral_grid，附相邻站链接及对应图节点编号 |
| ct_reference_topology.json.gz | 四个训练源的 1 mm 切片图，含独立分量、候选和断点原因 |
| blender_holdout_geometry_links.npz | 与原评估器 −510 至 −1280 mm 网格一致的几何中心和 154 条相邻站链接 |
| blender_holdout_topology.json.gz | 仅供评估的 Blender 具名图、审计和源文件哈希 |
| training_topology_audit.json | 源哈希、结果计数、阈值结果及与原实验断点的对照 |
| verification.json | 重算节点分量、边类型、CT 连续性和 Blender 旧几何中心的检查 |
| medseg_ct_route.png / open3d_vein_gap.png | 原生 CT 接触审计和真实图谱小断点修复图 |

**load_ct_training(path, shape=0)** 读取 31 层 CT 坐标。返回的 **stations_mm** 按远端到近端递增，**role_centers_xz_mm** 按「轴向层、角色、x/z」排列。第 0 形体为中立位。每层标签可用性见 **source_role_available**，缺失坐标为 NaN。调用方需要从近端开始时可反向读取。

**ct_geometry.npz** 的节点状态为 0 缺失、1 直接采样、2 插值。边状态是 0 断开、1 源样本关联、2 推断、3 经源几何或标注证实的接点。布尔 **links** 表示可否沿图投影通过，实际纳入哪种边由 **edge_status** 决定。几何行可非空而相邻边断开。

## 离线复现

运行要求为本地原项目代码、已挂载的 PEI_T7 和 EXT_DRIVE、NumPy、SciPy、trimesh、nibabel、libigl 和 matplotlib。在有 nibabel 的环境可移除下面的 PYTHONPATH 附加项。本机对应离线模块已位于 /tmp/right-leg-render-deps。所有命令在该目录运行。

    export PYTHONDONTWRITEBYTECODE=1
    export OPENBLAS_NUM_THREADS=1
    export PYTHONPATH="/tmp/right-leg-render-deps:${PYTHONPATH:-}"
    export MPLCONFIGDIR=/tmp/leg-topology-mpl
    python run_topology.py train
    python run_topology.py holdout
    python validate_topology.py

训练与留出各由独立进程启动。训练进程守卫阻断留出血管曲线、留出骨网格、留出回放和留出评估标签的打开请求。原始 CT 与已配准图谱只读，所有文件写入 --out。旧训练观测里短段的训练设定由原程序保存。31 层数据包提供新范围的源几何与标签接口。
