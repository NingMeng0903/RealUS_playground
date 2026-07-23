# LW100 导轨伺服 — 第三方调试文档

> 项目路径：`/media/camp/EXT_DRIVE/RealUS_playground/rm75_control`  
> 驱动代码：`rm75_control/hw/lw100/`  
> 最后更新：2026-07-23（§9 含完整 LW100 追赶源码快照）

---

## 0. 根因（已确认，电机已转）

**现象：** Modbus 写入全部成功、使能后轴锁死，但 CTRG / 速度指令不转。

**根因：** `FA-4` / `FA-14` / `FD-0` 等模式参数写进去可以立刻读回，但**不会立刻生效**。必须：

1. 写 `FA-60=1` **软复位**，或断电重启；
2. 再使能 + CTRG。

实机验证（增量 +1 圈 @ 60 r/min）：

```
peak |speed|=60 r/min (monitor 0x1000)
```

驱动已自动在模式变更后执行 FA-60 软复位并重连。

**实时速度监视：** holding reg `0x1000` = 电机转速 r/min（有符号，实机 probe，手册未写）。

---

## 1. 硬件与网络

| 设备 | IP | 说明 |
|------|-----|------|
| PC | `192.168.1.80` + `192.168.0.80` | 双网口，经交换机分别连 RM75 / USR |
| RM75 机械臂 | `192.168.1.18` | 与 LW100 无关，同交换机 |
| USR-TCP232-304 | `192.168.0.7:8234` | RS485 透明 TCP Server |
| LW100 伺服 | Modbus 从站 `FA71=1` | CN3/CN4 RS485，1610 丝杆 10 mm/rev |

**USR 必须配置：**

- Work Mode = **TCP Server**（出厂默认是 Client → PC 连 8234 会 `Connection refused`）
- Local Port = **8234**
- Serial 与 LW100 `FA72`/`FA73` 一致（出厂 **9600 8N2**；提速后 **115200 8N1**）

**RS485 接线（CN3 或 CN4）：**

- USR A+ → pin5 (RS485+)
- USR B- → pin4 (RS485-)
- USR GND → pin7

协议：**Modbus RTU + CRC16**，经 TCP 透传，**不是** Modbus-TCP（无 MBAP 头）。

---

## 2. 控制模式（LW100 内部位置 Pr）

LW100 **没有** Modbus 周期同步位置（CSP）模式。数字孪生/导轨控制走 **内部位置寄存器 P1** + **DI 强制触发 CTRG**。

| 参数 | 值 | 含义 |
|------|-----|------|
| FA-4 | 0 | 位置控制模式 |
| FA-14 | 3 | 内部位置输入 |
| FD-0 | 0 / 1 | 0=绝对坐标；1=**增量**（调试推荐 1） |
| FD-2 | ±30000 | P1 圈数（**电机圈**，不是 mm） |
| FD-3 | ±max cnt | P1 圈内脉冲（FA-11 默认 10000 cnt/rev） |
| FD-4 | 0–5000 | P1 段速度 r/min |
| FA-53 | 1 | 软件强制使能（纯通信、CN1 未接 SON 时用） |
| FC-15 bit0 | 1 | 强制 SON（与 FA-53 双保险） |
| FC-18 | 见下 | POS2~0 选段 + CTRG 上升沿 |

**P1 触发（POS2 POS1 POS0 CTRG → 参数）：**

| POS2 | POS1 | POS0 | CTRG | 段 | 参数 |
|------|------|------|------|-----|------|
| 0 | 0 | 0 | ↑ | P1 | FD-2, FD-3, FD-4 |

**FC-18 位映射（手册 §7.2.4）：**

| Bit | 功能 |
|-----|------|
| 3 | CTRG（0x08） |
| 4 | POS0 |
| 5 | POS1 |
| 6 | POS2 |

P1 = POS 全 0，仅 Bit3 产生上升沿 → `FC-18`: `0 → 0x08 → 0`

**1610 丝杆换算：**

```
travel_mm = motor_revs × lead_mm     (lead = 10 mm/rev, 直连 1:1)
motor_revs = travel_mm / 10
```

例：`FD-2=1` → 电机 1 圈 → 滑块 **10 mm**（不是 1 mm）。

---

## 3. Modbus 寄存器地址（实机 LW100-400W 已 probe）

| 组 | 基址 | 例 | 十进制地址 |
|----|------|-----|-----------|
| FA-n | 0 | FA-71 | 71 |
| FC-n | 256 | FC-15 | 271 |
| FC-n | 256 | FC-18 | 274 |
| FC-n | 256 | FC-13/14 | 269/270 |
| FD-n | 512 | FD-0 | 512 |
| FD-n | 512 | FD-2/3/4 | 514/515/516 |

**注意：holding reg 100 可读但不可写为 FD-2，勿把 FD 基址当成 100。**

### 3.1 常见误读（第三方审查已确认）

| 寄存器 | 实际用途 | **不是** |
|--------|----------|----------|
| FC-13 / FC-14 | 设定当前位置坐标低/高 16 位 | 实时编码器位置 |
| FD-0=0 + FD-2=5 | 运动到绝对内部坐标 5 圈 | 「再转 5 圈」 |

**实时状态只能看驱动器面板监视（dp--）：**

| 面板 | 含义 |
|------|------|
| d-Pos | 当前位置 |
| d-CPos | 位置指令 |
| d-EPos | 位置偏差 |
| d-Cnt | 当前控制方式 |
| d-rn | 运行状态 |
| d-Err | 报警代码 |

Modbus 参数可读 FA-4/14/20/53、FD-0 等，**暂无已验证的 Modbus 实时位置寄存器**。

---

## 4. 驱动代码时序（`drive.py` 当前实现）

运动序列（`move_inc_mm` / `move_abs_mm` 共用）：

```
1. disable          (FC-15/18=0, FA-53=0)
2. configure        (FA4=0, FA14=3, FD-0=0|1)
3. FA-60=1 soft reset + TCP 重连   ← 模式首次生效关键！
4. write P1         (FD-2, FD-3, FD-4)
5. enable           (FA-53=1, FC-15 SON=1)
6. wait 1.0 s       (ZSFD 稳定)
7. CTRG rising edge (FC-18: 0 → 0x08 → 0，每步 0.2 s)
8. poll 0x1000      (peak |speed| 证明在转)
```

同进程内模式未变时跳过步骤 3（`_active_mode` 缓存）。

常量：

- `ENABLE_SETTLE_S = 1.0`
- `CTRG_EDGE_HOLD_S = 0.2`
- `SOFT_RESET_RECONNECT_S = 1.5`

API：

| 方法 | FD-0 | 含义 |
|------|------|------|
| `move_inc_mm(travel_mm)` | 1 | 增量，每次触发加 `travel_mm` |
| `move_abs_mm(target_mm)` | 0 | 绝对坐标；目标=当前则不动 |
| `soft_reset()` | — | FA-60=1 + 重连 |
| `enable_and_settle()` | — | 使能 + 等待 |
| `read_speed_rpm()` | — | 读 `0x1000` 实时转速 |
| `read_status()` | — | 含 FA-20/53 + `speed_rpm` |

---

## 5. 测试流程（推荐顺序）

### 0. 环境与通信

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground/rm75_control
source env.sh

# TCP 端口
nc -zv 192.168.0.7 8234

# Modbus probe（应 FA71=1，FD@512 FC@256）
python apps/lw100_rail_demo.py --diagnose
```

首次提速串口：

```bash
python apps/lw100_setup_115200.py
# 断电重启 LW100 后：
python apps/lw100_setup_115200.py --verify-only
```

### 1. 测试 enable（轴锁死判断）

使能 OK 的标准：**FA-53=1 后手拧轴应锁死**。锁死但命令不转 → 软件触发问题；能自由拧 → 未真正通电流。

```bash
timeout 25 python3 - <<'PY'
import time
from rm75_control.hw.lw100.modbus_rtu_tcp import ModbusRtuTcpClient, ModbusRtuTcpConfig, ModbusRtuError
cfg = ModbusRtuTcpConfig(host="192.168.0.7", port=8234, slave_id=1, timeout_s=2.0, retries=2)
FA53, FC15 = 53, 271
with ModbusRtuTcpClient(cfg) as c:
    c.write_register(4, 0)
    c.write_register(14, 3)
    c.write_register(FA53, 1)
    c.write_register(FC15, 1)
    print("ENABLED (FA-53=1, SON=1). Turn shaft by hand for 15s.", flush=True)
    print("Locked = enable OK. Free = no torque.", flush=True)
    for i in range(15):
        time.sleep(1)
        try:
            fa20 = c.read_holding_registers(20, 1)[0]
            print(f"  t={i+1}s FA-20={fa20} (1=ignore CWL/CCWL)", flush=True)
        except ModbusRtuError as e:
            print(f"  t={i+1}s read fail {e}", flush=True)
    c.write_register(FC15, 0)
    c.write_register(FA53, 0)
    print("DISABLED.", flush=True)
PY
```

### 2. 最小增量转动（**推荐第一个运动测试**）

```bash
# 封装脚本（推荐）
python apps/lw100_min_test.py --run --revs 1 --speed-rpm 60 -v

# 或等价 inline 脚本
python3 - <<'PY'
import time
from rm75_control.hw.lw100.modbus_rtu_tcp import ModbusRtuTcpClient, ModbusRtuTcpConfig, ModbusRtuError
cfg = ModbusRtuTcpConfig(host="192.168.0.7", port=8234, slave_id=1, timeout_s=2.0, retries=2)
FA4, FA14, FA53 = 4, 14, 53
FD0, FD2, FD3, FD4 = 512, 514, 515, 516
FC18 = 274
CTRG = 0x08

def w(c, a, v, l=""):
    c.write_register(a, v & 0xFFFF)
    rb = c.read_holding_registers(a, 1)[0]
    print(f"  OK {l} @{a}={v} rb={rb}", flush=True)

with ModbusRtuTcpClient(cfg) as c:
    w(c, FA53, 0, "disable")
    w(c, FC18, 0, "clear CTRG/POS")
    w(c, FA4, 0, "position mode")
    w(c, FA14, 3, "internal position mode")
    w(c, FD0, 1, "incremental mode")
    w(c, FD2, 1, "P1 = +1 revolution")
    w(c, FD3, 0, "P1 pulse remainder")
    w(c, FD4, 60, "P1 speed = 60 rpm")
    print("FA-20 =", c.read_holding_registers(20, 1)[0], flush=True)
    w(c, FA53, 1, "software enable")
    time.sleep(1.0)
    w(c, FC18, 0x00, "CTRG low")
    time.sleep(0.2)
    w(c, FC18, CTRG, "CTRG rising")
    time.sleep(0.2)
    w(c, FC18, 0x00, "CTRG low")
    print("Watch panel: d-CPos then d-Pos", flush=True)
    time.sleep(4.0)
    w(c, FA53, 0, "disable")
print("done", flush=True)
PY
```

### 3. 导轨 demo（增量 / 绝对）

```bash
# 增量 +5 mm（推荐）
python apps/lw100_rail_demo.py --run --incremental --move-mm 5 -v

# 绝对坐标 5 mm（仅当当前不在 5 mm 处才有动作）
python apps/lw100_rail_demo.py --run --move-mm 5 --return -v
```

---

## 6. 故障排查

### 6.1 通信

| 现象 | 处理 |
|------|------|
| `Connection refused` @ 8234 | USR 改 TCP **Server** |
| Modbus timeout | USR 波特率/校验与 FA72/FA73 不一致 |
| FA71 probe 失败 | RS485 A/B 反接、从站号、LW100 未上电 |

### 6.2 使能 vs 运动

| 现象 | 判断 |
|------|------|
| 轴锁死，写参数成功，但不转 | **缺 FA-60 软复位**（模式未激活） |
| 轴锁死，d-CPos 不变 | CTRG/FA-14/HOLD/时序问题 |
| 轴锁死，d-CPos 变，d-Pos 不变 | 机械被阻（抱闸/卡死/禁止/转矩） |
| 轴可自由拧 | 未使能：查 FA-53、主回路 L1/L2、报警 |
| `peak \|speed\|=0` | 命令未执行；先确认软复位日志存在 |

### 6.3 报警与参数

| 代码/参数 | 含义 |
|-----------|------|
| Err 7 | 正反转驱动禁止（CWL/CCWL 均 OFF 且 FA-20=0） |
| Err 19 | 抱闸未释放时收到位置命令 |
| FA-20=0 | CWL/CCWL 禁止输入生效（出厂通常 =1 忽略） |
| 型号带 **B** | 机械抱闸，需独立 24V 释放 |

### 6.4 已知软件坑（已修）

| 旧行为 | 现状 |
|--------|------|
| 读 FC-13/14 当位置反馈 | 用 `0x1000` 转速 + 面板 d-Pos |
| FD-0=0 写 5 圈当「走 5 圈」 | 增量模式 `move_inc_mm` / `--incremental` |
| 使能后 0.1 s 就 CTRG | 固定等 1.0 s + 0.2 s 边沿 |
| 日志 `CTRG bit1` | 已改 `bit3` (0x08) |
| **模式写后立刻 CTRG** | **FA-60 软复位后再运动** |

---

## 7. Digital Twin 后续（未实现）

目标：高频率发位置指令 + 较低频率读真实位置（~30 Hz）。

**现状限制：**

- LW100 Modbus 仅支持 **Pr 内部位置 + CTRG 边沿触发**，非 CSP 流式位置；
- Modbus 上 **无已验证实时位置寄存器**，回读需面板监视或进一步手册/示波器确认是否有未文档化监视区；
- 每条运动 = 写 FD-2/3/4 + 使能等待 + CTRG，Modbus RTU 带宽上限约数十 Hz。

**可能路线：**

1. 短期：增量模式 + 主机维护绝对坐标估计，30 Hz 读面板/未来找到的监视寄存器；
2. 中期：双缓冲 P1/P2 交替 CTRG，提高有效指令率；
3. 长期：若需真正 CSP，需换支持 EtherCAT/CANopen CSP 的驱动器。

---

## 8. 相关文件

| 文件 | 用途 |
|------|------|
| `apps/lw100_min_test.py` | 最小增量测试 |
| `apps/lw100_rail_demo.py` | 完整 demo（`--diagnose` / `--incremental`） |
| `apps/lw100_setup_115200.py` | USR+LW100 串口提速 |
| `rm75_control/hw/lw100/drive.py` | 驱动 API |
| `rm75_control/hw/lw100/registers.py` | 寄存器映射 |
| `rm75_control/hw/lw100/README.md` | 网络/串口说明 |
| **§9 完整源码** | 当前 LW100 追赶 `q_cmd[0]` 全部相关代码（逐字复制） |

---

## 9. 完整源码快照 — LW100 导轨追赶 Controller `q_cmd[0]` (2026-07-23)

以下为 **逐字复制** 的当前仓库源码，供第三方审阅 smooth follow 实现。

**数据流：** WBC `inner.q_cmd[0]` → `RailServoBridge.set_target_m()` → 后台 worker（`follow_mode=position` Pr-P1 分段 + look-ahead，或 `follow_mode=velocity` FA24 软位置环）→ LW100 Modbus。
编码器只进 twin/SHM 与 velocity 闭环，**不**反馈进 WBC（见 `loop.py` `_rail_m_for_feedback`）。

### 文件索引

- `rm75_control/rm75_control/control/joint_admittance_8dof/hw/rail_servo.py`
- `rm75_control/rm75_control/hw/lw100/drive.py`
- `rm75_control/rm75_control/hw/lw100/modbus_rtu_tcp.py`
- `rm75_control/rm75_control/hw/lw100/registers.py`
- `rm75_control/rm75_control/hw/lw100/geometry.py`
- `rm75_control/rm75_control/hw/lw100/__init__.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/hw/__init__.py`
- `rm75_control/apps/joint_admittance_8dof/run_joint_admittance.py`
- `rm75_control/configs/joint_admittance_8dof.yaml`（`hw.lw100` 段，348–396 行）
- `rm75_control/rm75_control/control/joint_admittance_8dof/loop.py` L420–L569 — JointIkController.update — plan_drives_rail / rail pin (WBC 侧 q_cmd[0] 规划)
- `rm75_control/rm75_control/control/joint_admittance_8dof/loop.py` L1144–L1161 — loop.py — _rail_m_for_init / _rail_m_for_feedback
- `rm75_control/rm75_control/control/joint_admittance_8dof/loop.py` L1308–L1320 — loop.py — phase start q0 rail seed
- `rm75_control/rm75_control/control/joint_admittance_8dof/loop.py` L1366–L1372 — loop.py — phase enter q_meas rail
- `rm75_control/rm75_control/control/joint_admittance_8dof/loop.py` L1412–L1420 — loop.py — tick q_meas rail
- `rm75_control/rm75_control/control/joint_admittance_8dof/loop.py` L1477–L1486 — loop.py — 每 tick rail_bridge.set_target_m(q_cmd[0])
- `rm75_control/rm75_control/control/joint_admittance_8dof/loop.py` L1544–L1565 — loop.py — scan debug log rail cmd/meas
- `rm75_control/rm75_control/control/admittance_common/phase_ipc.py` L165–L170 — phase_ipc.py — request_stop (Ctrl-C 停 task)

### `rm75_control/rm75_control/control/joint_admittance_8dof/hw/rail_servo.py`

```py
"""LW100 rail servo bridge: track WBC ``q_cmd[0]`` on the physical rail.

Design:
  * WBC remains the sole planner — ``q_cmd[0]`` is streamed here.
  * ``follow_mode="position"`` (default, safer): Pr P1 incremental segments.
    Bounded open-loop steps inside ``[0, travel]``; no live FA24 hold.
  * ``follow_mode="velocity"``: speed mode (FA4=1) + soft position loop on FA24.
    Smooth when healthy, but FA24 latches if Modbus stalls — use only with
    encoder-range panic + estop wired to Ctrl-C.
  * Encoder is for SHM/twin / velocity loop only; never fed into the WBC.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from rm75_control.hw.lw100.drive import LW100Drive, LW100DriveConfig
from rm75_control.hw.lw100.modbus_rtu_tcp import ModbusRtuError


@dataclass
class RailServoConfig:
    enabled: bool = False
    host: str = "192.168.0.7"
    port: int = 8234
    slave_id: int = 1
    lead_mm: float = 10.0
    # "current": rail_y=0 at start pose (manual pre-home). "fixed": use counts0.
    zero_mode: str = "current"
    counts0: int = 0
    sign: float = 1.0
    enable_settle_s: float = 0.2
    poll_hz: float = 100.0
    deadband_mm: float = 0.5
    # Cap segment speed so motor matches controller rail v_max (0.20 m/s → 1200 r/min @ 10 mm/rev).
    max_speed_rpm: int = 1200
    busy_speed_rpm: int = 1
    # Follow mode: "position" = Pr P1 incremental segments (default, safer);
    #              "velocity" = continuous FA24 speed-mode servo (needs estop).
    follow_mode: str = "position"
    # Encoder outside [-margin, travel+margin] → panic (FA24=0, follow off).
    fault_margin_m: float = 0.05
    # Velocity-follow soft position loop (rail metres):
    vel_kp: float = 8.0          # 1/s: v_cmd = kp*(target-measured) + v_ff
    vel_ff_gain: float = 1.0     # feedforward fraction of target velocity
    vel_max_m_s: float = 0.20    # clamp on commanded rail speed (matches inner.rail.v_max)
    vel_deadband_mm: float = 0.3 # inside this → command 0 rpm (no dither)
    accel_ms: int = 15           # FA40 accel time (short = snappy continuous follow)
    decel_ms: int = 15           # FA41 decel time
    scurve_ms: int = 10          # FA42 S-curve time
    # Pr P1 does accel/decel per CTRG. Tiny segments → audible stop-start ("一卡一卡").
    # Look-ahead coalesces the WBC ramp into one/few long continuous segments.
    preview_s: float = 3.0
    # Wait until |target-commanded| reaches this before the first segment on a ramp
    # (avoids firing 1–2 mm crumbs while the quintic is still near zero velocity).
    commit_mm: float = 30.0
    # Hard cap on one segment (default = full travel). Keep ≥ travel for smooth moves.
    max_segment_mm: float = 800.0
    min_segment_mm: float = 1.0
    travel_m: float = 0.80
    timeout_s: float = 0.15
    retries: int = 1
    inter_frame_delay_s: float = 0.002
    home_on_exit: bool = False
    # Home cruise speed (r/min). 1200 r/min @ 10 mm/rev = 0.20 m/s = 20 cm/s.
    home_speed_rpm: int = 1200
    # Within this distance of 0, let kp*err (already below cruise) finish the stop.
    home_approach_mm: float = 40.0
    home_timeout_s: float = 60.0
    verbose: bool = False


def parse_rail_servo_config(raw: dict) -> RailServoConfig:
    """Build ``RailServoConfig`` from joint admittance YAML (``hw.lw100``)."""
    hw = raw.get("hw", {}).get("lw100", {}) or {}
    rail = raw.get("inner", {}).get("rail", {}) or {}
    travel_m = float(rail.get("travel_m", 0.80))
    # Default motor rpm from rail v_max: rpm = v(m/s) * 1000 / lead_mm * 60.
    lead_mm = float(hw.get("lead_mm", 10.0))
    v_max = float(rail.get("v_max_m_s", 0.20))
    default_rpm = max(60, int(round(v_max * 1000.0 / max(lead_mm, 1e-6) * 60.0)))
    zero_mode = str(hw.get("zero_mode", "current")).strip().lower()
    if zero_mode not in ("current", "fixed"):
        zero_mode = "current"
    follow_mode = str(hw.get("follow_mode", "position")).strip().lower()
    if follow_mode not in ("velocity", "position"):
        follow_mode = "position"
    return RailServoConfig(
        enabled=bool(hw.get("enabled", False)),
        host=str(hw.get("host", "192.168.0.7")),
        port=int(hw.get("port", 8234)),
        slave_id=int(hw.get("slave", hw.get("slave_id", 1))),
        lead_mm=lead_mm,
        zero_mode=zero_mode,
        counts0=int(hw.get("counts0", 0)),
        sign=float(hw.get("sign", 1.0)),
        enable_settle_s=float(hw.get("enable_settle_s", 0.2)),
        poll_hz=float(hw.get("poll_hz", 100.0)),
        deadband_mm=float(hw.get("deadband_mm", 0.5)),
        max_speed_rpm=int(hw.get("max_speed_rpm", default_rpm)),
        busy_speed_rpm=int(hw.get("busy_speed_rpm", 1)),
        follow_mode=follow_mode,
        fault_margin_m=float(hw.get("fault_margin_m", 0.05)),
        vel_kp=float(hw.get("vel_kp", 8.0)),
        vel_ff_gain=float(hw.get("vel_ff_gain", 1.0)),
        vel_max_m_s=float(hw.get("vel_max_m_s", v_max)),
        vel_deadband_mm=float(hw.get("vel_deadband_mm", 0.3)),
        accel_ms=int(hw.get("accel_ms", 15)),
        decel_ms=int(hw.get("decel_ms", 15)),
        scurve_ms=int(hw.get("scurve_ms", 10)),
        preview_s=float(hw.get("preview_s", 3.0)),
        commit_mm=float(hw.get("commit_mm", 30.0)),
        max_segment_mm=float(hw.get("max_segment_mm", travel_m * 1000.0)),
        min_segment_mm=float(hw.get("min_segment_mm", 1.0)),
        travel_m=travel_m,
        timeout_s=float(hw.get("timeout_s", 0.15)),
        retries=int(hw.get("retries", 1)),
        inter_frame_delay_s=float(hw.get("inter_frame_delay_s", 0.002)),
        home_on_exit=bool(hw.get("home_on_exit", False)),
        home_speed_rpm=int(hw.get("home_speed_rpm", default_rpm)),
        home_approach_mm=float(hw.get("home_approach_mm", 40.0)),
        home_timeout_s=float(hw.get("home_timeout_s", 60.0)),
        verbose=bool(hw.get("verbose", False)),
    )


class RailServoBridge:
    """Open-loop LW100 tracker: command stream → motor, encoder → twin only.

    Default workflow (no limit switches):
      * Start: treat current encoder pose as ``rail_y = 0`` (operator pre-homes manually).
      * Exit: open-loop command back to ``rail_y = 0``, then disable.
    """

    def __init__(self, config: RailServoConfig) -> None:
        self.config = config
        self.enabled = bool(config.enabled)
        self._target_m = 0.0
        self._commanded_m = 0.0  # rail-frame position already issued to the drive
        self._measured_m = 0.0
        self._lock = threading.Lock()
        self._drive: LW100Drive | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._segment_ready_mono = 0.0
        # False until WBC posts a target — prevents startup clamp/chase twitch.
        self._follow_enabled = False
        self._speed_cap_rpm: int | None = None
        # Target velocity EMA for look-ahead segment coalescing.
        self._tgt_v_m_s = 0.0
        self._last_tgt_m = 0.0
        self._last_tgt_mono = 0.0
        self._velocity_mode = config.follow_mode == "velocity"
        self._panic = False
        self._abort = threading.Event()

    @property
    def measured_m(self) -> float:
        with self._lock:
            return float(self._measured_m)

    @property
    def commanded_m(self) -> float:
        with self._lock:
            return float(self._commanded_m)

    @property
    def panicked(self) -> bool:
        with self._lock:
            return bool(self._panic)

    def set_target_m(self, target_m: float) -> None:
        """Host target in metres (WBC ``q_cmd[0]``). Clamped to [0, travel]; enables follow."""
        with self._lock:
            if self._panic:
                # Sticky only while encoder is still out of band.
                travel = float(self.config.travel_m)
                margin = max(float(self.config.fault_margin_m), 0.0)
                meas = float(self._measured_m)
                if not (-margin <= meas <= travel + margin):
                    return
                self._panic = False
            self._target_m = self._clamp_target_m(target_m)
            self._follow_enabled = True

    def hold_current(self) -> None:
        """Stop issuing new motion; freeze command stream at last commanded pose."""
        with self._lock:
            self._target_m = float(self._commanded_m)
            self._follow_enabled = False
        self.kill_motion()

    def kill_motion(self) -> None:
        """Immediate FA24=0 (velocity mode only). Safe from a signal handler."""
        drive = self._drive
        if drive is None or not self._velocity_mode:
            return
        try:
            drive.set_velocity_rpm(0, force=True)
        except Exception:
            try:
                drive.stop_velocity()
            except Exception:
                pass

    def estop(self) -> None:
        """Emergency stop: kill velocity, disable follow, abort home. No disable/close."""
        self._abort.set()
        with self._lock:
            self._follow_enabled = False
        self.kill_motion()

    def _encoder_sane(self, measured_m: float | None = None) -> bool:
        meas = float(self.measured_m if measured_m is None else measured_m)
        travel = float(self.config.travel_m)
        margin = max(float(self.config.fault_margin_m), 0.0)
        return -margin <= meas <= travel + margin

    def _trip_panic(self, measured: float, reason: str) -> None:
        with self._lock:
            already = self._panic
            self._panic = True
            self._follow_enabled = False
        self.kill_motion()
        if not already:
            print(
                f"lw100 rail: PANIC — {reason} "
                f"(meas={measured * 1000:.1f} mm, travel={self.config.travel_m * 1000:.0f} mm). "
                f"FA24=0, follow off. Fix encoder/limits before re-enable.",
                flush=True,
            )

    def _clamp_target_m(self, target_m: float) -> float:
        travel = float(self.config.travel_m)
        return max(0.0, min(travel, float(target_m)))

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        drive_cfg = LW100DriveConfig(
            host=self.config.host,
            port=self.config.port,
            slave_id=self.config.slave_id,
            timeout_s=self.config.timeout_s,
            retries=self.config.retries,
            inter_frame_delay_s=self.config.inter_frame_delay_s,
            lead_mm=self.config.lead_mm,
            enable_settle_s=self.config.enable_settle_s,
            verbose=self.config.verbose,
        )
        self._drive = LW100Drive(drive_cfg)
        self._drive.connect()
        self._velocity_mode = self.config.follow_mode == "velocity"
        if self._velocity_mode:
            self._drive.start_velocity_session(
                accel_ms=self.config.accel_ms,
                decel_ms=self.config.decel_ms,
                scurve_ms=self.config.scurve_ms,
            )
        else:
            self._drive.start_position_session(incremental=True)
            # Best-effort: clear leftover P1 distance (do not fail bring-up on a blip).
            try:
                self._drive.clear_p1_command()
            except ModbusRtuError as exc:
                print(f"lw100 rail: WARN clear P1: {exc}", flush=True)

        if self.config.zero_mode == "fixed":
            counts0 = int(self.config.counts0)
            self._drive.set_rail_zero(counts0)
            zero_note = f"fixed counts0={counts0}"
        else:
            # Operator has manually pre-homed; current pose is rail_y = 0.
            counts0 = int(self._drive.set_rail_zero())
            zero_note = f"current-as-zero counts0={counts0}"

        measured = float(self._drive.read_rail_m())
        raw = self._drive._read_encoder_counts_raw()
        with self._lock:
            self._measured_m = measured
            self._commanded_m = measured
            self._target_m = measured
            self._follow_enabled = False
            self._panic = False
            self._speed_cap_rpm = None
            self._segment_ready_mono = 0.0
            self._tgt_v_m_s = 0.0
            self._last_tgt_m = measured
            self._last_tgt_mono = time.monotonic()
        self._stop.clear()
        self._abort.clear()
        worker = self._worker_velocity if self._velocity_mode else self._worker
        self._thread = threading.Thread(target=worker, name="lw100-rail", daemon=True)
        self._thread.start()
        mode_note = (
            f"velocity-follow (kp={self.config.vel_kp}, "
            f"v_max={self.config.vel_max_m_s:.2f} m/s, poll={self.config.poll_hz:.0f}Hz, "
            f"modbus_gap={self.config.inter_frame_delay_s*1000:.1f}ms, "
            f"FA40/41={self.config.accel_ms}ms)"
            if self._velocity_mode
            else "open-loop Pr-P1 follow"
        )
        print(
            f"lw100 rail: hold @ {measured:+.4f} m ({zero_note}, "
            f"raw={raw} bias={self._drive._counts_bias}, "
            f"travel=[0, {self.config.travel_m:.2f}] m, "
            f"{mode_note}, home_on_exit={self.config.home_on_exit})",
            flush=True,
        )

    def go_home(self, *, timeout_s: float | None = None) -> bool:
        """Command ``rail_y -> 0``. Returns True if encoder reports arrival.

        Aborts immediately on ``estop()`` / second Ctrl-C. Refuses to chase if
        the encoder is already outside the travel band (avoids ±5 m runaways).
        """
        if not self.enabled or self._drive is None:
            return True
        if self._thread is None or not self._thread.is_alive():
            return abs(self.measured_m) * 1000.0 <= float(self.config.deadband_mm)

        meas0 = self.measured_m
        if not self._encoder_sane(meas0):
            print(
                f"lw100 rail: SKIP home — encoder out of range "
                f"(meas={meas0 * 1000:.1f} mm)",
                flush=True,
            )
            self.kill_motion()
            return False

        timeout = float(self.config.home_timeout_s if timeout_s is None else timeout_s)
        # Allow follow after a prior estop, but never after a range panic
        # that left the encoder insane (already handled above).
        with self._lock:
            self._panic = False
            self._speed_cap_rpm = int(self.config.home_speed_rpm)
        self._abort.clear()
        self.set_target_m(0.0)
        print(
            f"lw100 rail: homing to 0 (timeout={timeout:.0f}s, "
            f"cruise≤{self.config.home_speed_rpm} r/min "
            f"≈{self.config.home_speed_rpm / 60.0 * self.config.lead_mm / 10.0:.1f} cm/s, "
            f"approach={self.config.home_approach_mm:.0f} mm)…",
            flush=True,
        )
        deadband_m = float(self.config.deadband_mm) * 1e-3
        deadline = time.monotonic() + max(0.5, timeout)
        ok = False
        last_log = 0.0
        while time.monotonic() < deadline:
            if self._abort.is_set() or self._stop.is_set():
                self.kill_motion()
                with self._lock:
                    self._follow_enabled = False
                    self._speed_cap_rpm = None
                print("lw100 rail: home ABORTED", flush=True)
                return False
            meas = self.measured_m
            if not self._encoder_sane(meas):
                self._trip_panic(meas, "encoder left travel band during home")
                with self._lock:
                    self._speed_cap_rpm = None
                return False
            cmd = self.commanded_m
            try:
                busy = self._drive.is_busy(speed_threshold_rpm=self.config.busy_speed_rpm)
            except ModbusRtuError:
                busy = True
            if abs(meas) <= deadband_m and not busy:
                ok = True
                break
            if abs(cmd) <= deadband_m and abs(meas) <= 5.0 * deadband_m and not busy:
                ok = True
                break
            now = time.monotonic()
            if now - last_log >= 2.0:
                last_log = now
                print(
                    f"lw100 rail: home… meas={meas*1000:.1f} mm cmd={cmd*1000:.1f} mm "
                    f"busy={busy}",
                    flush=True,
                )
            time.sleep(0.05)
        self.hold_current()
        with self._lock:
            self._speed_cap_rpm = None
        print(
            f"lw100 rail: home {'OK' if ok else 'TIMEOUT'} @ {self.measured_m:+.4f} m "
            f"(cmd={self.commanded_m:+.4f} m)",
            flush=True,
        )
        return ok

    def stop(self, *, home: bool | None = None) -> None:
        """Stop the rail worker. Always kills live velocity first; home is optional.

        Order matters for safety:
          1) FA24=0 + follow off (never leave a latched speed command)
          2) optional home only if encoder is inside travel ± margin
          3) join worker, disable drive, close Modbus
        """
        self.kill_motion()
        with self._lock:
            self._follow_enabled = False

        do_home = self.config.home_on_exit if home is None else bool(home)
        if do_home and self._drive is not None and self._thread is not None:
            if self._encoder_sane():
                try:
                    self.go_home()
                except Exception as exc:
                    print(f"lw100 rail: WARN home on exit failed: {exc}", flush=True)
                    self.kill_motion()
            else:
                print(
                    f"lw100 rail: SKIP home on exit — encoder out of range "
                    f"(meas={self.measured_m * 1000:.1f} mm); disabling only",
                    flush=True,
                )

        self._abort.set()
        self._stop.set()
        self.kill_motion()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._drive is not None:
            try:
                self._drive.disable()
            except Exception:
                pass
            try:
                self._drive.close()
            except Exception:
                pass
            self._drive = None

    def _mps_to_rpm(self, v_m_s: float) -> float:
        """Rail linear speed (m/s) → motor r/min (lead mm/rev, direct drive)."""
        lead = max(float(self.config.lead_mm), 1e-6)
        return float(v_m_s) * 1000.0 / lead * 60.0

    def _worker_velocity(self) -> None:
        """Continuous velocity-follow: soft position loop → live FA24 (r/min).

        v_cmd = clamp( kp*(target - measured) + ff*target_vel, ±v_max )
        Encoder outside travel ± fault_margin → panic (FA24=0, follow off).
        Modbus errors also force FA24=0 so a stalled write cannot leave speed latched.
        """
        assert self._drive is not None
        period = 1.0 / max(float(self.config.poll_hz), 1.0)
        deadband_m = max(float(self.config.vel_deadband_mm), 0.05) * 1e-3
        v_max = float(self.config.vel_max_m_s)
        kp = float(self.config.vel_kp)
        ff = float(self.config.vel_ff_gain)
        sign = float(self.config.sign)
        travel = float(self.config.travel_m)
        margin = max(float(self.config.fault_margin_m), 0.0)
        prev_target: float | None = None
        prev_t = time.monotonic()
        v_ff = 0.0
        loop_n = 0
        loop_t0 = time.monotonic()
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                measured = float(self._drive.read_rail_m_fast())
                with self._lock:
                    self._measured_m = measured
                    self._commanded_m = measured  # velocity mode: truth = encoder
                    target = float(self._target_m)
                    follow = bool(self._follow_enabled)
                    panic = bool(self._panic)
                    speed_cap = self._speed_cap_rpm

                if measured < -margin or measured > travel + margin:
                    self._trip_panic(measured, "encoder outside travel band")
                    panic = True
                    follow = False

                # Target-velocity feedforward (EMA of dtarget/dt).
                if prev_target is not None:
                    dt = t0 - prev_t
                    if dt > 1e-4:
                        v_inst = (target - prev_target) / dt
                        v_ff = 0.5 * v_ff + 0.5 * v_inst
                prev_target = target
                prev_t = t0

                if panic or self._abort.is_set():
                    self._drive.set_velocity_rpm(0, force=True)
                elif follow:
                    err = target - measured
                    if abs(err) <= deadband_m:
                        v_cmd = 0.0
                    else:
                        v_cmd = kp * err + ff * v_ff
                    # End-stop guard: never drive further past travel limits.
                    if measured <= 0.0 and v_cmd < 0.0:
                        v_cmd = 0.0
                    elif measured >= travel and v_cmd > 0.0:
                        v_cmd = 0.0
                    v_lim = v_max
                    if speed_cap is not None:
                        # Home cruise, then linear taper in the approach band.
                        rpm_per_mps = max(abs(self._mps_to_rpm(1.0)), 1e-6)
                        cruise_m_s = abs(float(speed_cap)) / rpm_per_mps
                        approach_m = max(float(self.config.home_approach_mm), 1.0) * 1e-3
                        if abs(err) >= approach_m:
                            v_lim = min(v_lim, cruise_m_s)
                        else:
                            v_lim = min(v_lim, cruise_m_s * (abs(err) / approach_m))
                    v_cmd = max(-v_lim, min(v_lim, v_cmd))
                    rpm = sign * self._mps_to_rpm(v_cmd)
                    self._drive.set_velocity_rpm(rpm)
                    if self.config.verbose and abs(rpm) > 1.0:
                        print(
                            f"lw100 rail: v_follow tgt={target*1000:.1f} "
                            f"meas={measured*1000:.1f} mm err={err*1000:+.1f} "
                            f"v={v_cmd:+.3f} m/s → {rpm:+.0f} r/min",
                            flush=True,
                        )
                else:
                    # Not following: ensure motor is commanded to hold (0 speed).
                    if self._drive._last_rpm_cmd != 0:
                        self._drive.set_velocity_rpm(0)
                loop_n += 1
                if t0 - loop_t0 >= 2.0:
                    hz = loop_n / max(t0 - loop_t0, 1e-6)
                    print(
                        f"lw100 rail: loop {hz:.0f} Hz "
                        f"(tgt={target*1000:.1f} meas={measured*1000:.1f} mm "
                        f"follow={follow}{' PANIC' if panic else ''})",
                        flush=True,
                    )
                    loop_n = 0
                    loop_t0 = t0
            except ModbusRtuError as exc:
                # Never leave last FA24 active through a comms hole.
                try:
                    self._drive.set_velocity_rpm(0, force=True)
                except Exception:
                    pass
                if self.config.verbose:
                    print(f"lw100 rail: modbus error: {exc}", flush=True)
            except Exception as exc:
                try:
                    self._drive.set_velocity_rpm(0, force=True)
                except Exception:
                    pass
                if self.config.verbose:
                    print(f"lw100 rail: worker error: {exc}", flush=True)
            elapsed = time.monotonic() - t0
            sleep_s = max(0.0, period - elapsed)
            if self._stop.wait(sleep_s):
                break
        # Worker exit: one last zero so disable() is not the only brake.
        try:
            self._drive.set_velocity_rpm(0, force=True)
        except Exception:
            pass

    def _segment_time_s(self, step_mm: float, speed_rpm: int) -> float:
        """Estimate segment duration (motion + CTRG overhead)."""
        lead = max(float(self.config.lead_mm), 1e-6)
        revs = abs(float(step_mm)) / lead
        motion = (revs / max(float(speed_rpm), 1.0)) * 60.0
        # trigger_p1 uses ~2×20 ms holds after CTRG edge tune.
        return max(0.03, motion * 1.15 + 0.05)

    def _aim_m(self, target: float, now: float) -> float:
        """Look-ahead aim so a ramping WBC target becomes one long Pr segment.

        Pr P1 profiles accel/cruise/decel on every CTRG. Issuing 20 mm chunks
        while waiting for speed=0 produces the stop-start feel. Aiming ``preview_s``
        ahead of the target velocity collapses move→D into ~1–2 continuous runs.
        """
        travel = float(self.config.travel_m)
        target = max(0.0, min(travel, float(target)))
        last_t = self._last_tgt_mono
        last_x = self._last_tgt_m
        if last_t > 0.0:
            dt = now - last_t
            if dt > 1e-3:
                v_inst = (target - last_x) / dt
                # Fast attack / moderate release so a steady ramp locks in quickly.
                alpha = 0.35
                self._tgt_v_m_s = (1.0 - alpha) * self._tgt_v_m_s + alpha * v_inst
        self._last_tgt_m = target
        self._last_tgt_mono = now

        v = float(self._tgt_v_m_s)
        preview = max(0.0, float(self.config.preview_s))
        # Settled / slow: go exactly to target (final approach, hold, home).
        if abs(v) < 0.01 or preview <= 0.0:
            return target
        aim = target + v * preview
        # Never aim opposite the live target relative to commanded direction of v.
        if v > 0.0:
            aim = max(aim, target)
        else:
            aim = min(aim, target)
        return max(0.0, min(travel, aim))

    def _worker(self) -> None:
        assert self._drive is not None
        period = 1.0 / max(float(self.config.poll_hz), 1.0)
        deadband_m = float(self.config.deadband_mm) * 1e-3
        min_seg_m = max(float(self.config.min_segment_mm), 0.1) * 1e-3
        max_seg_m = max(float(self.config.max_segment_mm), 1.0) * 1e-3
        sign = float(self.config.sign)
        travel = float(self.config.travel_m)
        margin = max(float(self.config.fault_margin_m), 0.0)
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                measured = float(self._drive.read_rail_m_fast())
                with self._lock:
                    self._measured_m = measured
                    target = float(self._target_m)
                    commanded = float(self._commanded_m)
                    follow = bool(self._follow_enabled)
                    panic = bool(self._panic)
                    speed_cap = self._speed_cap_rpm

                if measured < -margin or measured > travel + margin:
                    self._trip_panic(measured, "encoder outside travel band")
                    follow = False

                if follow and (not panic) and (not self._abort.is_set()) and t0 >= self._segment_ready_mono:
                    try:
                        busy = self._drive.is_busy(
                            speed_threshold_rpm=self.config.busy_speed_rpm
                        )
                    except ModbusRtuError:
                        busy = True
                    # Open-loop + look-ahead: one long segment toward aim, not
                    # N×20 mm stop-starts. commanded tracks issued endpoints only.
                    aim = self._aim_m(target, t0)
                    delta_m = aim - commanded
                    # Also accept a final exact-target correction when settled.
                    if abs(delta_m) < deadband_m:
                        delta_m = target - commanded
                    err_to_tgt = target - commanded
                    settled = abs(self._tgt_v_m_s) < 0.01
                    commit_m = max(float(self.config.commit_mm), 1.0) * 1e-3
                    # On a ramp, wait until enough error accumulates so look-ahead
                    # can fire one long segment (not a crumb every poll).
                    if (
                        (not settled)
                        and abs(err_to_tgt) < commit_m
                        and abs(delta_m) < commit_m
                    ):
                        delta_m = 0.0
                    if (not busy) and abs(delta_m) >= max(deadband_m, min_seg_m):
                        step_m = max(-max_seg_m, min(max_seg_m, delta_m))
                        # Never command past software travel from the open-loop book.
                        next_cmd = commanded + step_m
                        next_cmd = max(0.0, min(travel, next_cmd))
                        step_m = next_cmd - commanded
                        if abs(step_m) < max(deadband_m, min_seg_m):
                            pass
                        else:
                            step_mm = step_m * 1000.0
                            motor_mm = sign * step_mm
                            cap = int(
                                self.config.max_speed_rpm
                                if speed_cap is None
                                else speed_cap
                            )
                            speed = max(60, min(cap, int(self.config.max_speed_rpm)))
                            if speed_cap is not None:
                                speed = max(60, min(cap, speed))
                            self._drive.command_inc_mm(motor_mm, speed_rpm=speed)
                            with self._lock:
                                self._commanded_m = commanded + step_m
                            self._segment_ready_mono = t0 + self._segment_time_s(step_mm, speed)
                            if self.config.verbose:
                                print(
                                    f"lw100 rail: seg {step_mm:+.1f} mm → cmd="
                                    f"{(commanded + step_m)*1000:.1f} mm "
                                    f"tgt={target*1000:.1f} aim={aim*1000:.1f} "
                                    f"meas={measured*1000:.1f} mm @{speed} r/min",
                                    flush=True,
                                )
            except ModbusRtuError as exc:
                if self.config.verbose:
                    print(f"lw100 rail: modbus error: {exc}", flush=True)
            except Exception as exc:
                if self.config.verbose:
                    print(f"lw100 rail: worker error: {exc}", flush=True)
            elapsed = time.monotonic() - t0
            sleep_s = max(0.0, period - elapsed)
            if self._stop.wait(sleep_s):
                break
```

### `rm75_control/rm75_control/hw/lw100/drive.py`

```py
"""High-level LW100 internal absolute position moves over Modbus RTU/TCP."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rm75_control.hw.lw100.geometry import PositionCommand, mm_to_position_command
from rm75_control.hw.lw100.modbus_rtu_tcp import ModbusRtuTcpClient, ModbusRtuTcpConfig, ModbusRtuError
from rm75_control.hw.lw100.registers import (
    ENCODER_COUNTS_PER_REV_17BIT,
    MONITOR_POS_HI,
    MONITOR_POS_LO,
    MONITOR_SPEED_RPM,
    P_FA11_PPR,
    P_FA14_POS_INPUT,
    P_FA20_DRIVE_INHIBIT,
    P_FA22_SPEED_SRC,
    P_FA24_INT_SPEED1,
    P_FA25_INT_SPEED2,
    P_FA26_INT_SPEED3,
    P_FA27_INT_SPEED4,
    P_FA40_ACC_MS,
    P_FA41_DEC_MS,
    P_FA42_SCURVE_MS,
    P_FA4_MODE,
    P_FA53_FORCE_ENABLE,
    P_FA60_SOFT_RESET,
    P_FA72_BAUD,
    P_FA73_PROTO,
    P_FC15_DI_FORCE1,
    P_FC18_DI_FORCE4,
    P_FD0_ABS_INC,
    P_FD2_P1_REVS,
    P_FD3_P1_PULSES,
    P_FD4_P1_SPEED,
    ParamRef,
    RegisterMap,
    probe_register_map,
)


# FC-15 (DI force 1) bit map: Bit0=SON per manual §7.2.4.
DI_SON = 1 << 0
# FC-18 (DI force 4) bit map per manual §7.2.4:
#   Bit3=CTRG, Bit4=POS0, Bit5=POS1, Bit6=POS2 (P1 = all POS low, pulse CTRG).
DI_CTRG = 1 << 3
DI_POS0 = 1 << 4
DI_POS1 = 1 << 5
DI_POS2 = 1 << 6

# FA72 = baud_rate / 100; FA73 per manual §7 (Modbus RTU format).
FA72_BAUD_9600 = 96
FA72_BAUD_115200 = 1152
FA73_PROTO_8N2 = 0
FA73_PROTO_8N1 = 3

# After FA-53 software enable, wait for ZSFD before accepting CTRG (manual FD-1 / CTRG §7.2).
ENABLE_SETTLE_S = 0.2
# Streaming follow needs short CTRG edges; 200 ms/edge limited the rail to ~2.5 Hz
# and made the twin/controller look stuttery. 20 ms is enough for the DI filter.
CTRG_EDGE_HOLD_S = 0.02
# FA4/FA14/FD-0 writes read back immediately but only become *active* after FA-60 soft reset
# (or power-cycle). Without this, enable/hold works but CTRG/speed commands are ignored.
SOFT_RESET_RECONNECT_S = 1.5


@dataclass
class LW100DriveConfig:
    host: str = "192.168.0.7"
    port: int = 8234
    slave_id: int = 1
    timeout_s: float = 0.15
    retries: int = 1
    inter_frame_delay_s: float = 0.002
    lead_mm: float = 10.0
    gear_ratio: float = 1.0
    pulses_per_rev: int = 10_000
    encoder_counts_per_rev: int = ENCODER_COUNTS_PER_REV_17BIT
    default_speed_rpm: int = 200
    configure_mode: bool = True
    enable_settle_s: float = ENABLE_SETTLE_S
    verbose: bool = False


@dataclass
class MoveResult:
    target_mm: float
    command: PositionCommand
    elapsed_s: float
    steps: list[str] = field(default_factory=list)


class LW100Drive:
    """LW100 rail driver: internal absolute position (Pr P1) via forced DI."""

    def __init__(self, config: LW100DriveConfig | None = None) -> None:
        self.config = config or LW100DriveConfig()
        self._client = ModbusRtuTcpClient(
            ModbusRtuTcpConfig(
                host=self.config.host,
                port=self.config.port,
                slave_id=self.config.slave_id,
                timeout_s=self.config.timeout_s,
                retries=self.config.retries,
                inter_frame_delay_s=self.config.inter_frame_delay_s,
            )
        )
        self._map: RegisterMap | None = None
        # Last mode tuple actually activated via FA-60 soft reset.
        self._active_mode: tuple[int, int, int] | None = None
        # Software home: rail_mm uses (counts - counts0). Call set_rail_zero() at machine home.
        self._counts0: int = 0
        # FA-60 soft-reset clears the drive's multi-turn monitor to ~0; bias keeps
        # host-side counts continuous across that wipe (not across power-loss).
        self._counts_bias: int = 0
        # True after start_position_session(); cleared on disable().
        self._position_session_active: bool = False
        # True after start_velocity_session(); cleared on disable().
        self._velocity_session_active: bool = False
        self._last_rpm_cmd: int = 0

    def connect(self) -> RegisterMap:
        self._client.connect()
        self._map = probe_register_map(
            self._client,
            expected_slave_id=self.config.slave_id,
            verbose=self.config.verbose,
        )
        if self.config.verbose:
            print(
                f"register map: FA@{self._map.bases['FA']} "
                f"FD@{self._map.bases['FD']} FC@{self._map.bases['FC']}",
                flush=True,
            )
        return self._map

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LW100Drive:
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.disable()
        except Exception:
            pass
        self.close()

    @property
    def register_map(self) -> RegisterMap:
        if self._map is None:
            raise RuntimeError("call connect() first")
        return self._map

    def _addr(self, param: ParamRef) -> int:
        return self.register_map.addr(param)

    def _log(self, steps: list[str], msg: str) -> None:
        steps.append(msg)
        if self.config.verbose:
            print(msg, flush=True)

    def read_param(self, param: ParamRef) -> int:
        vals = self._client.read_holding_registers(self._addr(param), 1)
        return int(vals[0])

    def write_param(self, param: ParamRef, value: int) -> None:
        self._client.write_register(self._addr(param), int(value))

    def read_pulses_per_rev(self) -> int:
        try:
            val = self.read_param(P_FA11_PPR)
            return val if val > 0 else self.config.pulses_per_rev
        except ModbusRtuError:
            return self.config.pulses_per_rev

    def soft_reset(self, steps: list[str] | None = None) -> None:
        """Pulse FA-60=1 so mode parameters (FA4/FA14/FD-0/…) become active.

        Live hardware fact: writes to FA4/FA14/FD-0 read back immediately, but
        motion commands (CTRG, internal speed) are ignored until soft-reset or
        power-cycle. TCP may drop briefly; we reconnect afterward.

        FA-60 also clears the encoder multi-turn monitor (~0). We snapshot counts
        before/after and accumulate ``_counts_bias`` so ``read_encoder_counts()``
        stays continuous for the host (power-cycle still loses multi-turn).
        """
        log = steps if steps is not None else []
        pre = 0
        try:
            pre = self._read_encoder_counts_raw()
        except ModbusRtuError:
            pass
        self._log(log, "FA-60=1 soft reset (activate mode params)")
        try:
            self.write_param(P_FA60_SOFT_RESET, 1)
        except ModbusRtuError as exc:
            self._log(log, f"WARN: FA-60 soft reset write failed: {exc}")
        time.sleep(SOFT_RESET_RECONNECT_S)
        try:
            self._client.close()
        except Exception:
            pass
        self._client.connect()
        self._log(log, "reconnected after soft reset")
        try:
            post = self._read_encoder_counts_raw()
            delta = int(pre) - int(post)
            if delta != 0:
                self._counts_bias += delta
                self._log(
                    log,
                    f"encoder bias += {delta} (pre={pre} post={post} bias={self._counts_bias})",
                )
        except ModbusRtuError as exc:
            self._log(log, f"WARN: encoder bias after soft reset failed: {exc}")

    def configure_internal_mode(
        self,
        *,
        incremental: bool = False,
        steps: list[str] | None = None,
        force_reset: bool = False,
    ) -> None:
        """Set FA4/FA14/FD-0 for internal position and soft-reset if mode changed."""
        log = steps if steps is not None else []
        if not self.config.configure_mode:
            self._log(log, "skip mode configure (configure_mode=False)")
            return
        fd0 = 1 if incremental else 0
        desired = (0, 3, fd0)  # FA4, FA14, FD-0
        # If drive already has the desired mode, skip FA-60 (avoids wiping multi-turn).
        if self._active_mode is None and not force_reset:
            try:
                cur = (
                    self.read_param(P_FA4_MODE),
                    self.read_param(P_FA14_POS_INPUT),
                    self.read_param(P_FD0_ABS_INC),
                )
                if cur == desired:
                    self._active_mode = desired
                    self._log(log, f"mode already live FA4/FA14/FD-0={desired} (no soft reset)")
            except ModbusRtuError:
                pass
        fd0_note = (
            "FD-0=1 incremental internal position"
            if incremental
            else "FD-0=0 absolute internal position"
        )
        writes = [
            (P_FA4_MODE, 0, "FA4=0 position mode"),
            (P_FA14_POS_INPUT, 3, "FA14=3 internal position input"),
            (P_FD0_ABS_INC, fd0, fd0_note),
        ]
        for param, value, note in writes:
            try:
                self.write_param(param, value)
                self._log(log, f"write {note} @ 0x{self._addr(param):04X}")
            except ModbusRtuError as exc:
                self._log(log, f"WARN: {note} failed: {exc}")

        if force_reset or self._active_mode != desired:
            self.soft_reset(log)
            # Re-assert mode after reset (params usually persist, belt-and-braces).
            for param, value, note in writes:
                try:
                    self.write_param(param, value)
                except ModbusRtuError:
                    pass
            self._active_mode = desired
            self._log(log, f"mode active FA4/FA14/FD-0={desired}")
        else:
            self._log(log, f"mode already active FA4/FA14/FD-0={desired}")

    def configure_internal_abs_mode(self, steps: list[str] | None = None) -> None:
        """Set FA4/FA14/FD-0 for internal absolute position (best-effort)."""
        self.configure_internal_mode(incremental=False, steps=steps)

    def configure_velocity_mode(
        self,
        *,
        accel_ms: int = 50,
        decel_ms: int = 50,
        scurve_ms: int = 20,
        steps: list[str] | None = None,
        force_reset: bool = False,
    ) -> None:
        """Set FA4=1 speed mode, FA22=1 internal speed (SP=00 → FA24), accel/decel ramps.

        Streaming FA24 (signed r/min) then gives continuous, smooth velocity
        following — the servo tracks a live velocity reference with no per-segment
        accel/decel stop-start (unlike Pr P1 point-to-point). Position is closed
        in software from the encoder (see ``RailServoBridge`` velocity follow).
        """
        log = steps if steps is not None else []
        if not self.config.configure_mode:
            self._log(log, "skip velocity mode configure (configure_mode=False)")
            return
        desired = (1, 1, 0)  # marker tuple for velocity mode (FA4, FA22, spare)
        if self._active_mode is None and not force_reset:
            try:
                cur_mode = self.read_param(P_FA4_MODE)
                cur_src = self.read_param(P_FA22_SPEED_SRC)
                if (cur_mode, cur_src) == (1, 1):
                    self._active_mode = desired
                    self._log(log, "velocity mode already live FA4=1 FA22=1 (no soft reset)")
            except ModbusRtuError:
                pass
        writes = [
            (P_FA4_MODE, 1, "FA4=1 speed control"),
            (P_FA22_SPEED_SRC, 1, "FA22=1 internal speed (SP selects FA24..27)"),
            # Factory FA25=500 / FA27=2000: if SP1/SP2 float high we must not leave
            # a non-zero cruise in unused slots (that caused smooth runaways to travel).
            (P_FA24_INT_SPEED1, 0, "FA24=0"),
            (P_FA25_INT_SPEED2, 0, "FA25=0"),
            (P_FA26_INT_SPEED3, 0, "FA26=0"),
            (P_FA27_INT_SPEED4, 0, "FA27=0"),
            (P_FA40_ACC_MS, int(accel_ms), f"FA40={accel_ms}ms accel"),
            (P_FA41_DEC_MS, int(decel_ms), f"FA41={decel_ms}ms decel"),
            (P_FA42_SCURVE_MS, int(scurve_ms), f"FA42={scurve_ms}ms S-curve"),
        ]
        for param, value, note in writes:
            try:
                self.write_param(param, value)
                self._log(log, f"write {note} @ 0x{self._addr(param):04X}")
            except ModbusRtuError as exc:
                self._log(log, f"WARN: {note} failed: {exc}")

        if force_reset or self._active_mode != desired:
            self.soft_reset(log)
            for param, value, note in writes:
                try:
                    self.write_param(param, value)
                except ModbusRtuError:
                    pass
            self._active_mode = desired
            self._log(log, "velocity mode active FA4=1 FA22=1")
        else:
            self._log(log, "velocity mode already active FA4=1 FA22=1")

    def start_velocity_session(
        self,
        *,
        accel_ms: int = 50,
        decel_ms: int = 50,
        scurve_ms: int = 20,
        steps: list[str] | None = None,
    ) -> None:
        """Configure speed mode once, enable, keep SON on for live FA24 streaming."""
        log = steps if steps is not None else []
        if self._velocity_session_active:
            self._log(log, "velocity session already active")
            return
        self.configure_velocity_mode(
            accel_ms=accel_ms, decel_ms=decel_ms, scurve_ms=scurve_ms, steps=log
        )
        self.enable_and_settle(log)
        self._last_rpm_cmd = 0
        self.set_velocity_rpm(0)
        self._velocity_session_active = True
        self._log(log, "velocity session started")

    def set_velocity_rpm(self, rpm: float, *, force: bool = False) -> int:
        """Write live velocity command FA24 (signed r/min, clamped ±6000).

        FA25..FA27 are zeroed at session start. SP1/SP2 default unmapped (OFF) so
        FA24 is the active slot; writing only FA24 is one Modbus transaction and
        keeps the follow loop near 50–100 Hz (writing all four slots dropped it
        to ~17 Hz and made motion feel chunky).

        Skips Modbus I/O when the command is unchanged.
        """
        r = int(max(-6000, min(6000, round(float(rpm)))))
        if (not force) and r == int(self._last_rpm_cmd):
            return r
        self.write_param(P_FA24_INT_SPEED1, r & 0xFFFF)
        self._last_rpm_cmd = r
        return r

    def stop_velocity(self) -> None:
        """Command zero velocity (best-effort)."""
        try:
            self.set_velocity_rpm(0)
        except ModbusRtuError:
            pass

    def enable(self, steps: list[str] | None = None) -> None:
        """Energize the motor for comms-only control.

        Uses FA-53=1 (software force enable) so no physical SON wiring on CN1 is
        needed, and also forces SON via FC-15 bit0 as a belt-and-braces measure.
        """
        log = steps if steps is not None else []
        try:
            self.write_param(P_FA53_FORCE_ENABLE, 1)
            self._log(log, "FA-53=1 software force enable")
        except ModbusRtuError as exc:
            self._log(log, f"WARN: FA-53 software enable failed: {exc}")
        try:
            self.write_param(P_FC15_DI_FORCE1, DI_SON)
            self._log(log, f"SON forced ON (FC-15 @ 0x{self._addr(P_FC15_DI_FORCE1):04X})")
        except ModbusRtuError as exc:
            self._log(log, f"WARN: FC-15 SON failed: {exc}")

    def enable_and_settle(self, steps: list[str] | None = None) -> None:
        """Enable, then wait for stable zero-speed before CTRG."""
        log = steps if steps is not None else []
        self.enable(log)
        dwell = max(0.0, float(self.config.enable_settle_s))
        if dwell > 0.0:
            self._log(log, f"wait {dwell:.1f}s after enable (ZSFD settle)")
            time.sleep(dwell)

    def disable(self, steps: list[str] | None = None) -> None:
        log = steps if steps is not None else []
        if self._velocity_session_active:
            try:
                self.set_velocity_rpm(0)
            except ModbusRtuError:
                pass
        try:
            self.write_param(P_FC15_DI_FORCE1, 0)
            self.write_param(P_FC18_DI_FORCE4, 0)
            self.write_param(P_FA53_FORCE_ENABLE, 0)
            self._log(log, "SON/CTRG released (FC-15/18=0, FA-53=0)")
        except ModbusRtuError:
            pass
        self._position_session_active = False
        self._velocity_session_active = False

    def start_position_session(
        self,
        *,
        incremental: bool = True,
        steps: list[str] | None = None,
    ) -> None:
        """Configure internal position once, enable, and keep SON on for segment commands.

        Subsequent moves use ``command_inc_mm`` / ``command_abs_mm`` without
        disable/soft-reset per segment.
        """
        log = steps if steps is not None else []
        if self._position_session_active:
            self._log(log, "position session already active")
            return
        self.configure_internal_mode(incremental=incremental, steps=log)
        self.enable_and_settle(log)
        self._position_session_active = True
        self._log(log, "position session started (incremental=%s)" % incremental)

    def command_inc_mm(
        self,
        travel_mm: float,
        *,
        speed_rpm: int | None = None,
        steps: list[str] | None = None,
    ) -> PositionCommand:
        """Fire one incremental P1 segment (requires ``start_position_session``)."""
        if not self._position_session_active:
            raise RuntimeError("call start_position_session() before command_inc_mm()")
        log = steps if steps is not None else []
        ppr = self.read_pulses_per_rev()
        cmd = mm_to_position_command(
            travel_mm,
            lead_mm=self.config.lead_mm,
            gear_ratio=self.config.gear_ratio,
            pulses_per_rev=ppr,
            speed_rpm=speed_rpm or self.config.default_speed_rpm,
        )
        self._write_p1_command(cmd, log)
        self.trigger_p1(log)
        return cmd

    def clear_p1_command(self, steps: list[str] | None = None) -> None:
        """Best-effort zero of P1 position fields (no CTRG).

        Speed is left alone: some drives NACK / time out on FD-4=0. Failures here
        must not abort session bring-up.
        """
        log = steps if steps is not None else []
        try:
            self.write_param(P_FD2_P1_REVS, 0)
            self.write_param(P_FD3_P1_PULSES, 0)
            self._log(log, "P1 cleared (rev=0 pulse=0, no CTRG)")
        except ModbusRtuError as exc:
            self._log(log, f"WARN: P1 clear failed: {exc}")

    def is_busy(self, *, speed_threshold_rpm: int = 1) -> bool:
        """True while the drive reports non-zero segment speed."""
        try:
            return abs(self.read_speed_rpm()) > int(speed_threshold_rpm)
        except ModbusRtuError:
            return True

    def _write_p1_command(self, cmd: PositionCommand, steps: list[str]) -> None:
        # Signed values: manual allows +/-30000 revs and +/-max cnt pulses.
        rev_val = int(cmd.revolutions) & 0xFFFF
        pulse_val = int(cmd.pulses) & 0xFFFF
        self.write_param(P_FD2_P1_REVS, rev_val)
        self.write_param(P_FD3_P1_PULSES, pulse_val)
        self.write_param(P_FD4_P1_SPEED, int(cmd.speed_rpm))
        self._log(
            steps,
            f"P1 target rev={cmd.revolutions} pulse={cmd.pulses} speed={cmd.speed_rpm} r/min",
        )

    def trigger_p1(self, steps: list[str] | None = None) -> None:
        """Select internal position P1 (POS=000) and pulse CTRG rising edge."""
        log = steps if steps is not None else []
        hold = max(0.005, float(CTRG_EDGE_HOLD_S))
        # POS2=0, POS1=0, POS0=0 — CTRG low
        self.write_param(P_FC18_DI_FORCE4, 0)
        time.sleep(hold)
        # CTRG rising edge (Bit3 = 0x08)
        self.write_param(P_FC18_DI_FORCE4, DI_CTRG)
        self._log(log, "CTRG rising edge (FC-18 bit3, P1 POS=000)")
        time.sleep(hold)
        self.write_param(P_FC18_DI_FORCE4, 0)

    def _execute_p1_move(
        self,
        cmd: PositionCommand,
        *,
        incremental: bool,
        steps: list[str],
        wait: bool,
    ) -> None:
        self.disable(steps)
        self.configure_internal_mode(incremental=incremental, steps=steps)
        self._write_p1_command(cmd, steps)
        self.enable_and_settle(steps)
        self.trigger_p1(steps)
        if wait:
            dwell = self.estimate_move_time_s(cmd)
            self._log(steps, f"wait {dwell:.1f}s for segment")
            # Poll live speed so logs prove motion (0x1000 monitor).
            t_end = time.monotonic() + dwell
            peak = 0
            while time.monotonic() < t_end:
                try:
                    rpm = abs(self.read_speed_rpm())
                    peak = max(peak, rpm)
                except ModbusRtuError:
                    pass
                time.sleep(0.2)
            self._log(steps, f"peak |speed|={peak} r/min (monitor 0x1000)")

    def estimate_move_time_s(self, cmd: PositionCommand) -> float:
        speed = max(float(cmd.speed_rpm), 1.0)
        revs = abs(float(cmd.revolutions)) + abs(float(cmd.pulses)) / float(
            max(self.config.pulses_per_rev, 1)
        )
        return max(0.5, (revs / speed) * 60.0 * 1.5)

    def move_abs_mm(
        self,
        target_mm: float,
        *,
        speed_rpm: int | None = None,
        wait: bool = True,
    ) -> MoveResult:
        """Move to absolute internal coordinate (FD-0=0).

        ``target_mm`` is the absolute screw coordinate from origin, not a delta.
        If the target equals the current coordinate the drive will not move.
        """
        steps: list[str] = []
        t0 = time.monotonic()
        ppr = self.read_pulses_per_rev()
        cmd = mm_to_position_command(
            target_mm,
            lead_mm=self.config.lead_mm,
            gear_ratio=self.config.gear_ratio,
            pulses_per_rev=ppr,
            speed_rpm=speed_rpm or self.config.default_speed_rpm,
        )
        self._execute_p1_move(cmd, incremental=False, steps=steps, wait=wait)
        elapsed = time.monotonic() - t0
        return MoveResult(target_mm=target_mm, command=cmd, elapsed_s=elapsed, steps=steps)

    def move_inc_mm(
        self,
        travel_mm: float,
        *,
        speed_rpm: int | None = None,
        wait: bool = True,
    ) -> MoveResult:
        """Move by signed delta (FD-0=1 incremental). Each trigger adds ``travel_mm``."""
        steps: list[str] = []
        t0 = time.monotonic()
        ppr = self.read_pulses_per_rev()
        cmd = mm_to_position_command(
            travel_mm,
            lead_mm=self.config.lead_mm,
            gear_ratio=self.config.gear_ratio,
            pulses_per_rev=ppr,
            speed_rpm=speed_rpm or self.config.default_speed_rpm,
        )
        self._execute_p1_move(cmd, incremental=True, steps=steps, wait=wait)
        elapsed = time.monotonic() - t0
        return MoveResult(target_mm=travel_mm, command=cmd, elapsed_s=elapsed, steps=steps)

    def stop(self) -> None:
        self.disable()

    def read_speed_rpm(self) -> int:
        """Live motor speed (r/min) from monitor register 0x1000 (signed)."""
        val = int(self._client.read_holding_registers(MONITOR_SPEED_RPM, 1)[0])
        return val - 0x10000 if val >= 0x8000 else val

    def _read_encoder_counts_raw(self, *, retries: int = 5) -> int:
        """Drive monitor 0x1001/0x1002 only (no host bias)."""
        last: tuple[int, int] | None = None
        for _ in range(max(1, retries)):
            lo, hi = self._client.read_holding_registers(MONITOR_POS_LO, 2)
            pair = (int(lo) & 0xFFFF, int(hi) & 0xFFFF)
            if last == pair:
                v = (pair[1] << 16) | pair[0]
                return v - (1 << 32) if v >= (1 << 31) else v
            last = pair
        assert last is not None
        v = (last[1] << 16) | last[0]
        return v - (1 << 32) if v >= (1 << 31) else v

    def read_encoder_counts(self, *, retries: int = 5) -> int:
        """Live encoder position as signed 32-bit counts (monitor 0x1001/0x1002).

        Live-proved at idle: +1 motor revolution → +``encoder_counts_per_rev``
        (131072 for 17-bit). Includes ``_counts_bias`` so FA-60 soft-reset does
        not jump the host-side position. Power-cycle still loses multi-turn on
        this drive (17-bit single-turn absolute class).
        """
        return self._read_encoder_counts_raw(retries=retries) + int(self._counts_bias)

    def set_rail_zero(self, counts: int | None = None) -> int:
        """Software-home the rail at the current (or given) encoder counts.

        ``counts`` must be in the same host frame as ``read_encoder_counts()``
        (includes bias). Fixed YAML ``counts0`` is only valid within one powered
        session unless the motor has battery-backed multi-turn absolute.
        """
        self._counts0 = int(self.read_encoder_counts() if counts is None else counts)
        return self._counts0

    def read_rail_mm(self) -> float:
        """Measured rail position in mm from encoder (never from command).

        ``rail_mm = (counts - counts0) / counts_per_rev * lead_mm / gear_ratio``
        """
        counts = float(self.read_encoder_counts() - self._counts0)
        cpr = float(max(self.config.encoder_counts_per_rev, 1))
        motor_revs = counts / cpr
        return (motor_revs / float(self.config.gear_ratio)) * float(self.config.lead_mm)

    def read_rail_m(self) -> float:
        """Measured rail position in metres (Genesis ``rail_y``)."""
        return self.read_rail_mm() * 1e-3

    def read_rail_m_fast(self) -> float:
        """Streaming rail position (metres): ONE Modbus transaction, no double-read.

        ``read_encoder_counts(retries=5)`` re-reads until two transactions agree.
        While the axis is moving they never agree, so it always burns 5 round-trips
        (~75–150 ms) → the poll loop drops to ~7–13 Hz and the twin stutters, worst
        exactly when the rail moves. lo/hi come back in a single Modbus response
        (no word-tear within a transaction), so a single read is safe for display
        and the soft position loop.
        """
        raw = self._read_encoder_counts_raw(retries=1) + int(self._counts_bias)
        counts = float(raw - self._counts0)
        cpr = float(max(self.config.encoder_counts_per_rev, 1))
        motor_revs = counts / cpr
        return (motor_revs / float(self.config.gear_ratio)) * float(self.config.lead_mm) * 1e-3

    def read_status(self) -> dict[str, int]:
        """Mode / enable params + live speed. Prefer ``read_rail_mm`` for position."""
        out: dict[str, int] = {}
        for param in (
            P_FA4_MODE,
            P_FA14_POS_INPUT,
            P_FA20_DRIVE_INHIBIT,
            P_FA53_FORCE_ENABLE,
            P_FD0_ABS_INC,
            P_FC15_DI_FORCE1,
            P_FA72_BAUD,
            P_FA73_PROTO,
        ):
            try:
                out[param.label] = self.read_param(param)
            except ModbusRtuError:
                out[param.label] = -1
        try:
            out["speed_rpm"] = self.read_speed_rpm()
        except ModbusRtuError:
            out["speed_rpm"] = -1
        try:
            out["encoder_counts"] = self.read_encoder_counts()
        except ModbusRtuError:
            out["encoder_counts"] = -1
        return out

    def setup_modbus_serial(
        self,
        *,
        fa72_baud_code: int = FA72_BAUD_115200,
        fa73_proto: int = FA73_PROTO_8N1,
    ) -> list[str]:
        """Write FA72/FA73 from the host (no drive keypad required).

        Connect at the *current* drive baud (factory 9600 8N2) via USR first,
        then call this, power-cycle the drive, and match USR to the new rate.
        """
        steps: list[str] = []
        before72 = self.read_param(P_FA72_BAUD)
        before73 = self.read_param(P_FA73_PROTO)
        self._log(steps, f"before: FA-72={before72} FA-73={before73}")
        self.write_param(P_FA72_BAUD, int(fa72_baud_code))
        self.write_param(P_FA73_PROTO, int(fa73_proto))
        after72 = self.read_param(P_FA72_BAUD)
        after73 = self.read_param(P_FA73_PROTO)
        self._log(
            steps,
            f"after:  FA-72={after72} ({after72 * 100} bps)  "
            f"FA-73={after73}  (3=8N1, 0=8N2)",
        )
        if after72 != int(fa72_baud_code) or after73 != int(fa73_proto):
            raise ModbusRtuError(
                f"FA72/FA73 readback mismatch: got {after72}/{after73}, "
                f"expected {fa72_baud_code}/{fa73_proto}"
            )
        self._log(
            steps,
            "NEXT: (1) power-cycle LW100  (2) USR -> 115200 8N1  "
            "(3) python apps/lw100_rail_demo.py --diagnose",
        )
        return steps
```

### `rm75_control/rm75_control/hw/lw100/modbus_rtu_tcp.py`

```py
"""Modbus RTU client over TCP transparent serial (USR-TCP232-304)."""

from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass


def crc16_modbus(data: bytes) -> int:
    """CRC-16/MODBUS (poly 0xA001, init 0xFFFF)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


def append_crc(frame: bytes) -> bytes:
    crc = crc16_modbus(frame)
    return frame + struct.pack("<H", crc)


def verify_crc(frame: bytes) -> bool:
    if len(frame) < 3:
        return False
    payload, crc_bytes = frame[:-2], frame[-2:]
    expected = struct.unpack("<H", crc_bytes)[0]
    return crc16_modbus(payload) == expected


class ModbusRtuError(RuntimeError):
    """Modbus exception or transport failure."""


@dataclass
class ModbusRtuTcpConfig:
    host: str
    port: int = 8234
    slave_id: int = 1
    timeout_s: float = 1.0
    retries: int = 2
    # At 115200, 3.5 RTU chars ≈ 0.3 ms. USR-TCP232 needs a few ms of turnaround.
    # 50 ms was historically used for flaky links but capped the rail loop at ~10 Hz
    # (read+write), which looked like one update per motor revolution in the twin.
    inter_frame_delay_s: float = 0.002


class ModbusRtuTcpClient:
    """Send Modbus RTU ADUs through a TCP serial server (no MBAP header)."""

    FC_READ_HOLDING = 0x03
    FC_WRITE_SINGLE = 0x06
    FC_WRITE_MULTIPLE = 0x10

    def __init__(self, config: ModbusRtuTcpConfig) -> None:
        self.config = config
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        if self._sock is not None:
            return
        sock = socket.create_connection(
            (self.config.host, self.config.port),
            timeout=self.config.timeout_s,
        )
        sock.settimeout(self.config.timeout_s)
        self._sock = sock

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __enter__(self) -> ModbusRtuTcpClient:
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _drain_rx(self) -> None:
        """Discard any stale bytes in the TCP receive buffer."""
        if self._sock is None:
            return
        self._sock.setblocking(False)
        try:
            while True:
                chunk = self._sock.recv(256)
                if not chunk:
                    break
        except (BlockingIOError, InterruptedError):
            pass
        finally:
            self._sock.setblocking(True)
            self._sock.settimeout(self.config.timeout_s)

    def send_raw(self, request: bytes) -> bytes:
        """Send RTU ADU (without CRC) and return raw bytes from TCP (diagnostics)."""
        if self._sock is None:
            raise ModbusRtuError("not connected")
        self._drain_rx()
        req = append_crc(request)
        self._sock.sendall(req)
        time.sleep(self.config.inter_frame_delay_s)
        return self._read_response_raw()

    def _send_receive(self, request: bytes) -> bytes:
        if self._sock is None:
            raise ModbusRtuError("not connected")
        req = append_crc(request)
        last_err: Exception | None = None
        for attempt in range(max(1, self.config.retries)):
            try:
                self._drain_rx()
                self._sock.sendall(req)
                time.sleep(self.config.inter_frame_delay_s)
                response = self._read_response()
                if not verify_crc(response):
                    raise ModbusRtuError(f"CRC mismatch on response: {response.hex()}")
                if response[0] != request[0]:
                    raise ModbusRtuError(
                        f"slave mismatch: sent id={request[0]}, got id={response[0]}"
                    )
                fc = response[1]
                if fc & 0x80:
                    exc_code = response[2] if len(response) > 2 else -1
                    raise ModbusRtuError(f"Modbus exception fc=0x{request[1]:02x} code={exc_code}")
                return response
            except (TimeoutError, socket.timeout, OSError, ModbusRtuError) as err:
                last_err = err
                if attempt + 1 < self.config.retries:
                    time.sleep(self.config.inter_frame_delay_s)
                    continue
                raise ModbusRtuError(str(last_err)) from last_err
        raise ModbusRtuError("unreachable")

    def _read_response_raw(self) -> bytes:
        assert self._sock is not None
        buf = bytearray()
        deadline = time.monotonic() + self.config.timeout_s
        while time.monotonic() < deadline:
            try:
                chunk = self._sock.recv(256)
            except socket.timeout:
                break
            if not chunk:
                break
            # Drop leading idle/noise nulls before a real ADU starts.
            if not buf:
                chunk = chunk.lstrip(b"\x00")
                if not chunk:
                    continue
            buf.extend(chunk)
            if len(buf) >= 5 and verify_crc(bytes(buf)):
                return bytes(buf)
            # Truncate leading noise if slave id never appears.
            if len(buf) > 64:
                buf.clear()
        return bytes(buf)

    def _read_response(self) -> bytes:
        buf = self._read_response_raw()
        if not buf:
            raise ModbusRtuError("response timeout")
        if not verify_crc(buf):
            raise ModbusRtuError(f"CRC mismatch on response: {buf.hex()}")
        return buf

    def read_holding_registers(self, address: int, count: int = 1) -> list[int]:
        addr = int(address) & 0xFFFF
        cnt = int(count) & 0xFFFF
        req = struct.pack(
            ">BBHH",
            self.config.slave_id,
            self.FC_READ_HOLDING,
            addr,
            cnt,
        )
        resp = self._send_receive(req)
        if resp[1] != self.FC_READ_HOLDING:
            raise ModbusRtuError(f"unexpected function code {resp[1]}")
        byte_count = resp[2]
        data = resp[3 : 3 + byte_count]
        if len(data) != byte_count or byte_count != 2 * count:
            raise ModbusRtuError(f"unexpected read length: {resp.hex()}")
        return list(struct.unpack(f">{count}H", data))

    def write_register(self, address: int, value: int) -> None:
        addr = int(address) & 0xFFFF
        val = int(value) & 0xFFFF
        req = struct.pack(
            ">BBHH",
            self.config.slave_id,
            self.FC_WRITE_SINGLE,
            addr,
            val,
        )
        resp = self._send_receive(req)
        if resp[1] != self.FC_WRITE_SINGLE:
            raise ModbusRtuError(f"unexpected function code {resp[1]}")

    def write_registers(self, address: int, values: list[int]) -> None:
        if not values:
            return
        addr = int(address) & 0xFFFF
        payload = b"".join(struct.pack(">H", int(v) & 0xFFFF) for v in values)
        req = struct.pack(
            ">BBHHB",
            self.config.slave_id,
            self.FC_WRITE_MULTIPLE,
            addr,
            len(values),
            len(payload),
        ) + payload
        resp = self._send_receive(req)
        if resp[1] != self.FC_WRITE_MULTIPLE:
            raise ModbusRtuError(f"unexpected function code {resp[1]}")
```

### `rm75_control/rm75_control/hw/lw100/registers.py`

```py
"""LW100 parameter → Modbus holding-register address mapping."""

from __future__ import annotations

from dataclasses import dataclass, field

from rm75_control.hw.lw100.modbus_rtu_tcp import ModbusRtuTcpClient, ModbusRtuError


@dataclass(frozen=True)
class ParamRef:
    group: str
    index: int

    @property
    def label(self) -> str:
        return f"{self.group.upper()}-{self.index}"


# Frequently used parameters (LW100 manual ch.7).
P_FA4_MODE = ParamRef("FA", 4)  # 0=position 1=speed 2=torque 3=pos/speed
# Speed-mode params (manual §5.2): FA22 speed-cmd source, FA24 internal speed 1 (signed r/min),
# FA40/41 accel/decel time (ms), FA42 S-curve time (ms).
P_FA22_SPEED_SRC = ParamRef("FA", 22)   # 1=internal speed by SP1/SP2 (SP=00 → FA24)
P_FA24_INT_SPEED1 = ParamRef("FA", 24)  # -6000..6000 r/min, live velocity command
P_FA25_INT_SPEED2 = ParamRef("FA", 25)
P_FA26_INT_SPEED3 = ParamRef("FA", 26)
P_FA27_INT_SPEED4 = ParamRef("FA", 27)
P_FA40_ACC_MS = ParamRef("FA", 40)
P_FA41_DEC_MS = ParamRef("FA", 41)
P_FA42_SCURVE_MS = ParamRef("FA", 42)
P_FA11_PPR = ParamRef("FA", 11)
P_FA14_POS_INPUT = ParamRef("FA", 14)
P_FA20_DRIVE_INHIBIT = ParamRef("FA", 20)  # 0=CWL/CCWL inhibit active, 1=ignore (factory 1)
P_FA53_FORCE_ENABLE = ParamRef("FA", 53)  # 0=SON via DI, 1=software force enable
P_FA60_SOFT_RESET = ParamRef("FA", 60)  # 1=soft reset (required after mode changes)
P_FA71_SLAVE = ParamRef("FA", 71)
P_FA72_BAUD = ParamRef("FA", 72)
P_FA73_PROTO = ParamRef("FA", 73)

# Undocumented monitor block (live-probed on LW100-400W):
#   0x1000        = motor speed (r/min, signed)
#   0x1001/0x1002 = encoder position int32 (lo, hi), +131072 counts per motor rev (17-bit)
#                   Stable at idle (span ≤2 counts). Prefer idle/double-read snapshots.
#   0x100C/0x100D = NOT reliable live position (noisy / word-tear); do not use.
MONITOR_SPEED_RPM = 0x1000
MONITOR_POS_LO = 0x1001
MONITOR_POS_HI = 0x1002
ENCODER_COUNTS_PER_REV_17BIT = 131_072
P_FC13_POS_COORD_LO = ParamRef("FC", 13)  # set current position coord low 16b — NOT live feedback
P_FC14_POS_COORD_HI = ParamRef("FC", 14)  # set current position coord high 16b — NOT live feedback
P_FC15_DI_FORCE1 = ParamRef("FC", 15)
# FC-16 bit1=SP1, bit2=SP2 (manual §7.2.4). SP selects FA24..FA27 in speed mode.
P_FC16_DI_FORCE2 = ParamRef("FC", 16)
P_FC18_DI_FORCE4 = ParamRef("FC", 18)
P_FD0_ABS_INC = ParamRef("FD", 0)
P_FD2_P1_REVS = ParamRef("FD", 2)
P_FD3_P1_PULSES = ParamRef("FD", 3)
P_FD4_P1_SPEED = ParamRef("FD", 4)


# Live hardware register map (confirmed on LW100-400W):
#   FA-n  → n            (e.g. FA71 @ 71)
#   FD-n  → 512 + n      (FD-0 @512, FD-2 @514, FD-3 @515, FD-4 @516)
#   FC-n  → 256 + n      (FC-15 @271, FC-18 @274)
# NOTE: holding reg 100 is a position-loop gain param, NOT FD-0. Do not use base 100.
DEFAULT_GROUP_BASE = {"FA": 0, "FD": 512, "FC": 256}


@dataclass
class RegisterMap:
    """Group base addresses for FA/FC/FD parameters."""

    bases: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_GROUP_BASE))

    def addr(self, param: ParamRef) -> int:
        g = param.group.upper()
        if g not in self.bases:
            raise ValueError(f"unknown parameter group {g!r}")
        return int(self.bases[g]) + int(param.index)


def _try_read(client: ModbusRtuTcpClient, addr: int) -> int | None:
    try:
        return int(client.read_holding_registers(addr, 1)[0])
    except ModbusRtuError:
        return None


def _try_write(client: ModbusRtuTcpClient, addr: int, value: int) -> bool:
    try:
        client.write_register(addr, int(value) & 0xFFFF)
        return True
    except ModbusRtuError:
        return False


def _writable(client: ModbusRtuTcpClient, addr: int, test_value: int) -> bool:
    old = _try_read(client, addr)
    if old is None:
        return False
    if not _try_write(client, addr, test_value):
        return False
    rb = _try_read(client, addr)
    _try_write(client, addr, old)
    return rb == (test_value & 0xFFFF)


def probe_register_map(
    client: ModbusRtuTcpClient,
    *,
    expected_slave_id: int = 1,
    verbose: bool = False,
) -> RegisterMap:
    """Probe FA71, then locate FD/FC bases on a live drive."""
    # FA is always parameter index.
    fa71 = _try_read(client, 71)
    if verbose:
        print(f"  probe FA71@71: {fa71}", flush=True)
    if fa71 != expected_slave_id:
        # legacy guesses
        for addr in (0xFA47, 70, 0x0147):
            val = _try_read(client, addr)
            if verbose:
                print(f"  probe alt FA71@{addr}: {val}", flush=True)
        raise ModbusRtuError(
            f"could not probe LW100 (FA71@{71} should read {expected_slave_id}, got {fa71}). "
            "Check USR serial matches drive (115200 8N1 after setup), RS485 A/B, power."
        )

    bases = dict(DEFAULT_GROUP_BASE)

    # FD position command block: FD-2 (revs) writable, FD-4 (speed) default ~1000.
    # Verify by write-restore on FD-2 so a coincidental read at another base is
    # rejected (holding reg 100 reads 0/1 but is NOT writable as FD-2).
    fd_base = None
    for cand in (512, 100, 200, 256, 0x0D00):
        fd2 = cand + 2
        val = _try_read(client, fd2)
        if verbose:
            print(f"  probe FD-2 cand @{fd2}: {val}", flush=True)
        if val is None:
            continue
        if _writable(client, fd2, val if val is not None else 0):
            fd_base = cand
            break
    if fd_base is not None:
        bases["FD"] = fd_base
        if verbose:
            print(f"  FD base = {fd_base} (FD-2@{fd_base + 2})", flush=True)

    # FC-15 (DI force 1 / SON). Default 0, must be writable.
    fd0 = bases["FD"]
    fc_base = None
    for cand_base in (256, 150, 200, 300, 0x0C00):
        if fd0 <= cand_base + 15 < fd0 + 60:
            continue
        addr = cand_base + 15
        val = _try_read(client, addr)
        if verbose:
            print(f"  probe FC-15 cand @{addr}: {val}", flush=True)
        if val is None:
            continue
        if _writable(client, addr, val):
            fc_base = cand_base
            break
    if fc_base is not None:
        bases["FC"] = fc_base
        if verbose:
            print(f"  FC base = {fc_base} (FC-15@{fc_base + 15})", flush=True)

    return RegisterMap(bases=bases)


def diagnose_bus(
    client: ModbusRtuTcpClient,
    *,
    slave_ids: tuple[int, ...] = (1,),
    verbose: bool = True,
) -> None:
    """Print raw Modbus probes (for serial / address troubleshooting)."""
    import struct

    from rm75_control.hw.lw100.modbus_rtu_tcp import append_crc

    if verbose:
        print(
            "\nRS485 checklist (no bytes = serial/wiring, not register map):\n"
            "  1) USR serial must match LW100 FA72/FA73 (factory: 9600 8N2; after setup: 115200 8N1)\n"
            "     TCP Server / port 8234\n"
            "  2) LW100 powered, no alarm; cable on CN3 or CN4 (RJ45)\n"
            "  3) USR A+ -> drive pin5 RS485+ ; USR B- -> drive pin4 RS485-\n"
            "  4) Common GND (USR GND -> drive pin7)\n"
            "  5) If still silent, swap A/B once; power-cycle drive\n",
            flush=True,
        )

    test_addrs = (71, 72, 100, 0xFA47)
    for sid in slave_ids:
        for addr in test_addrs:
            req = struct.pack(">BBHH", sid, 0x03, addr & 0xFFFF, 1)
            tx = append_crc(req)
            if verbose:
                print(f"  TX slave={sid} read 0x{addr:04X}: {tx.hex()}", flush=True)
            try:
                raw = client.send_raw(req)
            except ModbusRtuError as exc:
                if verbose:
                    print(f"       -> {exc}", flush=True)
                continue
            if not raw:
                if verbose:
                    print("       -> (no bytes)", flush=True)
                continue
            if verbose:
                print(f"       -> RX {len(raw)} bytes: {raw.hex()}", flush=True)
                return
```

### `rm75_control/rm75_control/hw/lw100/geometry.py`

```py
"""1610 ball-screw geometry: mm ↔ motor revolutions + instruction pulses."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PositionCommand:
    """Internal absolute position segment (Pr P1 registers)."""

    revolutions: int
    pulses: int
    speed_rpm: int


def mm_to_position_command(
    travel_mm: float,
    *,
    lead_mm: float = 10.0,
    gear_ratio: float = 1.0,
    pulses_per_rev: int = 10_000,
    speed_rpm: int = 200,
) -> PositionCommand:
    """Convert linear travel (mm) to LW100 internal position command fields.

    Parameters
    ----------
    travel_mm:
        Signed distance along the screw (+/-).
    lead_mm:
        Screw lead in mm/rev (1610 → 10 mm).
    gear_ratio:
        Motor revolutions per screw revolution (1.0 = direct coupling).
    pulses_per_rev:
        ``FA11`` — instruction pulses per motor revolution.
    speed_rpm:
        ``FD-4`` segment speed (r/min).
    """
    if lead_mm <= 0.0:
        raise ValueError(f"lead_mm must be > 0, got {lead_mm}")
    if gear_ratio <= 0.0:
        raise ValueError(f"gear_ratio must be > 0, got {gear_ratio}")
    if pulses_per_rev <= 0:
        raise ValueError(f"pulses_per_rev must be > 0, got {pulses_per_rev}")

    total_revs = (float(travel_mm) / lead_mm) * gear_ratio
    sign = 1 if total_revs >= 0.0 else -1
    abs_revs = abs(total_revs)
    whole = int(math.floor(abs_revs + 1e-12))
    frac = abs_revs - whole
    pulses = int(round(frac * pulses_per_rev))
    if pulses >= pulses_per_rev:
        whole += 1
        pulses = 0
    revolutions = sign * whole
    if sign < 0:
        pulses = -pulses if pulses else 0
    return PositionCommand(revolutions=revolutions, pulses=pulses, speed_rpm=int(speed_rpm))


def position_command_to_mm(
    cmd: PositionCommand,
    *,
    lead_mm: float = 10.0,
    gear_ratio: float = 1.0,
    pulses_per_rev: int = 10_000,
) -> float:
    """Inverse of ``mm_to_position_command`` (approximate for display)."""
    motor_revs = float(cmd.revolutions) + float(cmd.pulses) / float(pulses_per_rev)
    return (motor_revs / gear_ratio) * lead_mm
```

### `rm75_control/rm75_control/hw/lw100/__init__.py`

```py
"""LW100 servo over Modbus RTU via USR-TCP232 Ethernet-RS485 gateway."""

from rm75_control.hw.lw100.drive import LW100Drive, LW100DriveConfig
from rm75_control.hw.lw100.geometry import PositionCommand, mm_to_position_command

__all__ = [
    "LW100Drive",
    "LW100DriveConfig",
    "PositionCommand",
    "mm_to_position_command",
]
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/hw/__init__.py`

```py
"""Hardware bridges for joint_admittance_8dof (LW100 rail, etc.)."""

from rm75_control.control.joint_admittance_8dof.hw.rail_servo import (
    RailServoBridge,
    RailServoConfig,
    parse_rail_servo_config,
)

__all__ = [
    "RailServoBridge",
    "RailServoConfig",
    "parse_rail_servo_config",
]
```

### `rm75_control/apps/joint_admittance_8dof/run_joint_admittance.py`

```py
#!/usr/bin/env python3
"""8-DOF controller daemon (window A): UDP + SHM + local WBC when C submits a task.

Window A in the 3-terminal layout: keeps the sole Realman TCP/UDP session,
publishes ``rm75_state`` for the Genesis twin, and **runs the 200 Hz WBC loop
locally** when window C submits a phase program (no per-tick CANFD SHM relay).

  source env.sh
  python apps/joint_admittance_8dof/run_joint_admittance.py \\
      --config configs/joint_admittance_8dof.yaml

Twin (separate terminal):

  python apps/joint_admittance_8dof/run_with_twin.py

Task orchestration (window C):

  python apps/joint_admittance_8dof/d_sin_tool_y.py --config ... --enable-force ...
"""

from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path

import numpy as np
import yaml

from rm75_control.control.admittance_common.phase_ipc import PhaseCmd, PhaseCommandHub, PhaseStatus
from rm75_control.control.admittance_common.state_bus import RobotStateBus
from rm75_control.control.admittance_common.state_relay import (
    StateRelayPublisher,
    parse_state_relay_config,
)
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import (
    CartesianTrackConfig,
    CartesianTrackOuterLoop,
    JointIkController,
    run_joint_admittance_loop,
)
from rm75_control.control.joint_admittance_8dof.hw.rail_servo import (
    RailServoBridge,
    parse_rail_servo_config,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.reference import HoldReference
from rm75_control.control.joint_admittance_8dof.sin_tool_y_program import (
    build_sin_tool_y_program,
    execute_sin_tool_y_program,
)
from rm75_control.core.session import RobotSession
from rm75_control.force.compensation.tool_pose import maybe_sync_kin_tcp_from_config


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _run_controller_service(
    sess,
    bus: RobotStateBus,
    raw: dict,
    *,
    hub: PhaseCommandHub,
    rail_m_fn,
    rail_bridge: RailServoBridge | None = None,
    poll_s: float = 0.05,
    verbose: bool = False,
) -> None:
    """Hot-wait for window C; run WBC locally on START (direct UDP + CANFD)."""
    stop = False

    def _on_sig(_signum, _frame) -> None:
        nonlocal stop
        # First action: kill rail velocity so FA24 cannot stay latched while
        # teardown / home runs. Then request WBC stop so the task loop exits.
        if rail_bridge is not None and rail_bridge.enabled:
            rail_bridge.estop()
        try:
            hub.request_stop()
        except Exception:
            pass
        stop = True

    signal.signal(signal.SIGINT, _on_sig)
    signal.signal(signal.SIGTERM, _on_sig)

    hub.set_idle()
    print("rm75 controller: hot-wait", flush=True)

    while not stop:
        polled = hub.poll()
        if polled is None:
            time.sleep(poll_s)
            continue

        cmd, cmd_seq, params = polled
        if cmd == PhaseCmd.STOP:
            hub.ack(cmd_seq)
            hub.set_stopped(cmd_seq)
            continue

        if cmd != PhaseCmd.START or params is None:
            hub.ack(cmd_seq)
            continue

        task_n = hub.task_n
        hub.set_running(cmd_seq, msg="accepted")
        print(f"rm75 controller: running task #{task_n}", flush=True)
        phase_labels: list[str] = []
        tick_counter = [0]
        phase_idx = [0]
        last_progress_label = [""]

        def _on_step(label, t_phase, step, pose, f_ext, t_wall=float("nan")) -> None:
            tick_counter[0] += 1
            if label in phase_labels:
                idx = phase_labels.index(label)
            else:
                phase_labels.append(label)
                idx = len(phase_labels) - 1
            phase_idx[0] = idx
            label_s = str(label)
            if label_s != last_progress_label[0]:
                last_progress_label[0] = label_s
                hub.set_progress(
                    cmd_seq,
                    phase_idx=idx,
                    phase_label=label_s,
                    ticks=tick_counter[0],
                )

        try:
            built = build_sin_tool_y_program(params, raw=raw)
            rail_m_fn.set_active(built.inner)
            result = execute_sin_tool_y_program(
                sess,
                bus,
                params,
                raw=raw,
                built=built,
                on_step=_on_step,
                stop_check=hub.should_stop,
                verbose=verbose,
                rail_bridge=rail_bridge,
            )
            if hub.should_stop():
                hub.set_stopped(cmd_seq)
                print(f"rm75 controller: task #{task_n} stopped", flush=True)
            else:
                hub.set_done(cmd_seq)
                print(
                    f"rm75 controller: task #{task_n} done "
                    f"({result.duration_s:.1f}s, {result.ticks} ticks)",
                    flush=True,
                )
        except KeyboardInterrupt:
            stop = True
            hub.set_stopped(cmd_seq, msg="interrupted")
            print(f"rm75 controller: task #{task_n} interrupted", flush=True)
        except Exception as exc:
            hub.set_error(cmd_seq, str(exc))
            print(f"rm75 controller: task error: {exc}", flush=True)
        finally:
            hub.ack(cmd_seq)
            rail_m_fn.reset_idle()
            if rail_bridge is not None and rail_bridge.enabled:
                rail_bridge.hold_current()
            if not stop:
                print("rm75 controller: hot-wait", flush=True)


class _RailPublisher:
    """Mutable rail source for SHM twin during idle vs active WBC.

    When the LW100 bridge is enabled, publish **encoder** position (poll_hz)
    so the twin mirrors the real carriage. WBC itself uses open-loop ``q_cmd[0]``
    and does not close the loop on this value.
    """

    def __init__(self, default_m: float, bridge: RailServoBridge | None = None) -> None:
        self._default_m = float(default_m)
        self._bridge = bridge
        self._active_inner: JointIkController | None = None

    def reset_idle(self) -> None:
        if self._bridge is not None and self._bridge.enabled:
            self._default_m = float(self._bridge.measured_m)
        elif self._active_inner is not None:
            self._default_m = float(self._active_inner.q_cmd[0])
        self._active_inner = None

    def set_active(self, inner: JointIkController) -> None:
        self._active_inner = inner

    def __call__(self) -> float:
        if self._bridge is not None and self._bridge.enabled:
            return float(self._bridge.measured_m)
        if self._active_inner is not None:
            return float(self._active_inner.q_cmd[0])
        return self._default_m


def main() -> int:
    ap = argparse.ArgumentParser(
        description="8-DOF controller daemon (window A)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--config", type=Path, default=Path("configs/joint_admittance_8dof.yaml"))
    ap.add_argument(
        "--state-relay",
        default="rm75_state",
        metavar="NAME",
        help="Publish robot state to SHM for twin / window C (default rm75_state)",
    )
    ap.add_argument("--no-state-relay", action="store_true", help="Do not publish SHM")
    ap.add_argument("--relay-hz", type=float, default=None, help="SHM publish rate (default from YAML)")
    ap.add_argument(
        "--hold",
        action="store_true",
        help="Stream CANFD idle hold (teach re-anchor). Do NOT use with d_sin_tool_y.py",
    )
    ap.add_argument("--verbose", "-v", action="store_true", help="Print loop / teach / phase status")
    ap.add_argument("--dry-run", action="store_true", help="build controllers only, do not connect")
    args = ap.parse_args()

    raw = load_yaml(args.config)
    startup = raw.get("startup", {})
    relay_cfg = parse_state_relay_config(raw)
    if args.no_state_relay:
        relay_name = None
    else:
        relay_name = str(args.state_relay or relay_cfg.name or "rm75_state")
    relay_hz = float(args.relay_hz) if args.relay_hz is not None else relay_cfg.hz
    dt = float(raw.get("timing", {}).get("dt_ms", 5.0)) / 1000.0
    rail_default_m = float(raw.get("inner", {}).get("rail", {}).get("q_ref_m", 0.0))
    rail_bridge = RailServoBridge(parse_rail_servo_config(raw))
    rail_pub = _RailPublisher(rail_default_m, bridge=rail_bridge)

    if args.dry_run:
        mode = "hold+CANFD" if args.hold else "controller+hot-wait"
        print(f"rm75 controller: dry-run OK ({mode})", flush=True)
        return 0

    robot_cfg = raw.get("robot", {})
    relay: StateRelayPublisher | None = None
    inner: JointIkController | None = None
    hub: PhaseCommandHub | None = None

    if args.hold:
        kin = RobotKinematics()
        inner_cfg = build_joint_ik_config(raw)
        inner = JointIkController(kin, inner_cfg)
        rail_pub.set_active(inner)

    with RobotSession(
        ip=robot_cfg.get("ip"),
        port=robot_cfg.get("port"),
        config=args.config,
        quiet=True,
    ) as sess:
        try:
            if rail_bridge.enabled:
                rail_bridge.start()
                rail_pub._default_m = float(rail_bridge.measured_m)
            if inner is not None:
                maybe_sync_kin_tcp_from_config(raw=raw, kin=inner.kin, robot=sess.robot)
            else:
                maybe_sync_kin_tcp_from_config(
                    raw=raw, kin=RobotKinematics(), robot=sess.robot
                )
            bus = RobotStateBus(sess.robot, raw, robot_ip=sess.ip)
            bus.start()

            if relay_name:
                relay = StateRelayPublisher(
                    bus,
                    name=relay_name,
                    hz=relay_hz,
                    rail_m_fn=rail_pub,
                )
                relay.start()
                if args.hold:
                    print(
                        f"rm75 controller: hold @ {relay_hz:.0f} Hz",
                        flush=True,
                    )
                else:
                    print(
                        f"rm75 controller: running @ {relay_hz:.0f} Hz",
                        flush=True,
                    )
            elif args.hold:
                print("rm75 controller: hold (no SHM)", flush=True)
            else:
                print("rm75 controller: running (no SHM)", flush=True)

            if args.hold:
                assert inner is not None
                outer = CartesianTrackOuterLoop(
                    HoldReference(),
                    CartesianTrackConfig(
                        k_task=np.full(6, 2.0),
                        euler_order=inner.cfg.euler_order,
                        control_frame=inner.cfg.control_frame,
                    ),
                )
                run_joint_admittance_loop(
                    sess,
                    outer,
                    inner,
                    q_start_deg=None,
                    duration_s=None,
                    dt=dt,
                    force_observer=None,
                    follow=bool(startup.get("follow", True)),
                    move_speed=int(startup.get("move_speed", 20)),
                    realtime=bool(startup.get("realtime", False)),
                    watchdog_timeout_s=float(startup.get("watchdog_timeout_s", 0.1)),
                    state_bus=bus,
                    verbose=args.verbose,
                    rail_bridge=rail_bridge,
                )
            else:
                hub = PhaseCommandHub()
                _run_controller_service(
                    sess,
                    bus,
                    raw,
                    hub=hub,
                    rail_m_fn=rail_pub,
                    rail_bridge=rail_bridge,
                    verbose=args.verbose,
                )
        finally:
            if hub is not None:
                hub.close()
            if relay is not None:
                relay.stop()
            rail_bridge.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### `rm75_control/configs/joint_admittance_8dof.yaml`（hw.lw100 段，348–396 行）

```yaml
# LW100 rail servo (Modbus RTU over USR-TCP232).
# follow_mode=position (default, safer): Pr P1 incremental segments inside [0, travel].
# follow_mode=velocity: speed mode (FA4=1) + soft position loop on FA24. Smooth when
#   healthy, but FA24 latches if Modbus stalls — Ctrl-C calls estop (FA24=0) first;
#   encoder outside travel±fault_margin_m trips PANIC (FA24=0, follow off, skip home).
# Encoder is SHM/twin display (+ velocity loop); never fed into the WBC.
# Workflow without limit switches (default):
#   1) Manually push carriage to -Y end (mechanical 0) before starting controller.
#   2) zero_mode=current → start pose becomes rail_y=0.
#   3) home_on_exit=false (default) → on exit, estop + disable only; no auto return to 0.
# Software clamps targets to [0, inner.rail.travel_m].
hw:
  lw100:
    enabled: true
    host: 192.168.0.7
    port: 8234
    slave: 1
    lead_mm: 10.0
    zero_mode: current
    counts0: 0
    sign: 1
    enable_settle_s: 0.2
    follow_mode: position    # position (safer) | velocity (smooth, needs estop)
    fault_margin_m: 0.05     # encoder outside [-margin, travel+margin] → PANIC
    poll_hz: 100             # target control+feedback rate (needs low inter_frame_delay)
    inter_frame_delay_s: 0.002  # was 0.05 → capped loop at ~10 Hz ("一圈一更新")
    timeout_s: 0.15
    retries: 1
    deadband_mm: 0.5
    max_speed_rpm: 1200      # matches inner.rail.v_max_m_s=0.20 @ 10 mm/rev
    # velocity-follow soft position loop (only if follow_mode=velocity):
    vel_kp: 8.0              # 1/s
    vel_ff_gain: 1.0
    vel_max_m_s: 0.20        # = inner.rail.v_max_m_s
    vel_deadband_mm: 0.3
    accel_ms: 15             # FA40 (short for continuous follow)
    decel_ms: 15             # FA41
    scurve_ms: 10            # FA42
    # position-mode (follow_mode=position) knobs:
    preview_s: 3.0
    commit_mm: 30
    max_segment_mm: 800
    min_segment_mm: 1.0
    busy_speed_rpm: 1
    home_on_exit: false
    home_speed_rpm: 1200     # 20 cm/s cruise home
    home_approach_mm: 40     # linear taper to stop in last 40 mm
    home_timeout_s: 60
    verbose: false
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/loop.py` L420–L569

*JointIkController.update — plan_drives_rail / rail pin (WBC 侧 q_cmd[0] 规划)*

```python
        directly reassigning ``q_cmd`` bypasses the velocity/acceleration box
        and can command a multi-degree joint step in one tick (rm_movej_canfd
        treats that as a discontinuity - visible as violent shake/jerk on
        hardware). Some follow lag during a fast move is normal servo
        behaviour, not a fault; the QP bound just stops it from growing
        further, at the normal velocity-limited rate.  ``qdot_ff`` is a
        joint-space feedforward projected onto the task nullspace together
        with the centering / arm-angle tasks.
        """
        dt = self.cfg.dt if dt is None else dt
        q_prev = self.q_cmd
        follow_err = 0.0 if q_meas is None else float(np.max(np.abs(q_prev - q_meas)))
        q_rot = q_meas if q_meas is not None else q_prev
        twist_base = self._twist_to_base(twist, q_rot)

        # Rail mode dispatch (top-level + substyle)
        locked_hold = self.is_locked_hold
        rail_only = (
            self._rail_mode == RailMode.LOCKED
            and self._locked_style == LockedStyle.RAIL_ONLY
        )
        tcp_fixed = (
            self._rail_mode == RailMode.LOCKED
            and self._locked_style == LockedStyle.TCP_FIXED
        )
        # LOCKED styles always pin rail from the plan. Move phases also pass a
        # plan_* qdot_ff that includes rail (0→center); without pinning, COUPLED
        # Cartesian QP yanks rail across travel (hardware log: tgt 743→42→374 mm
        # on move->D) even though the joint plan is a smooth 0→0.4 m ramp.
        plan_drives_rail = rail_only or tcp_fixed or qdot_ff is not None

        qdot_ff_sec = qdot_ff
        rail_vel_pin: float | None = None
        if plan_drives_rail and qdot_ff is not None:
            qdot_ff_arr = np.asarray(qdot_ff, dtype=float)
            v_rail = float(qdot_ff_arr[0])
            # Strip rail out of the qdot_ff passed to the composer (secondary
            # tasks act only on the arm portion); the rail velocity is imposed
            # via the QP rail-vel pin below and then safety-clamped.
            qdot_ff_sec = qdot_ff_arr.copy()
            qdot_ff_sec[0] = 0.0
            rail_vel_pin = v_rail

        # Vectorized command-lead anti-windup: arm rad, rail m (units matter).
        resync_vec = np.full(self.kin.nv, float(self.cfg.resync_err_rad))
        resync_vec[0] = float(self.cfg.resync_err_rail_m)

        # Preferred-extension rail coordination (COUPLED only): the rail
        # proactively follows the TCP when the arm reaches past its
        # comfortable extension — early, smooth singularity avoidance
        # instead of reactive last-moment recruitment.
        rail_task_vel: float | None = None
        rail_task_weight = 0.0
        rail_ext_err = 0.0
        manip_for_saturation = self._manipulability_active
        if (
            self.rail_ext_task is not None
            and self._rail_ext_active
            and self._rail_mode == RailMode.COUPLED
        ):
            sigma_ref = float(self.cfg.qp.sr_damping.sigma_ref)
            sigma_now = float(
                self.kin.singular_values(self.kin.jacobian(q_prev)).min()
            )
            # Keep rail-extension authority below the (possibly softened)
            # Cartesian task so the QP never inverts priorities near σ dips.
            sig_scale = 1.0
            if sigma_ref > 1e-9 and sigma_now < sigma_ref:
                sig_scale = max(sigma_now / sigma_ref, 0.25)
            # Bug 2: refresh the σ-escape gradient every ``_sigma_grad_period``
            # ticks (analytical FD, cheap-but-not-free).  Passing 0 means the
            # σ-escape v-component collapses to the reach term (safe fallback).
            self._sigma_grad_tick += 1
            if (
                self._sigma_grad_tick % self._sigma_grad_period == 0
                or self._sigma_grad_tick == 1
            ):
                self._sigma_grad_rail_cached = sigma_min_grad_rail(self.kin, q_prev)
            v_ext, w_ext = self.rail_ext_task(
                q_prev,
                sigma_scale=sig_scale,
                sigma_grad_rail=self._sigma_grad_rail_cached,
                vel_ff=vel_ff,
            )
            rail_ext_err = self.rail_ext_task.last_err_m
            if w_ext > 0.0:
                rail_task_vel = v_ext
                rail_task_weight = w_ext
            elif self.rail_ext_task.last_limit_saturated and sigma_now < sigma_ref:
                # Rail cannot help further: escape arm singularities in the
                # nullspace instead of straightening the arm.
                manip_for_saturation = True

        r = self.core.step(
            q_prev,
            twist_base,
            dt,
            secondary_qdot=self._secondary(
                q_prev,
                qdot_ff_sec,
                manipulability_active=manip_for_saturation,
                centering_sigma_fade=not (
                    self._rail_ext_active and self._rail_mode == RailMode.COUPLED
                ),
            ),
            q_meas=q_meas,
            resync_err=resync_vec,
            rail_locked=locked_hold,
            rail_lock_reg_scale=self.cfg.rail.lock_reg_scale,
            rail_lock_vel_eps_m_s=self.cfg.rail.lock_vel_eps_m_s,
            rail_vel_pin_m_s=rail_vel_pin,
            zero_secondary_rail=not locked_hold,
            rail_task_vel_m_s=rail_task_vel,
            rail_task_weight=rail_task_weight,
        )

        rep = self.safety.clamp(q_prev, r.q_next, dt)
        self.q_cmd = rep.q_safe
        if dt > 1e-9 and (rep.vel_clamped or rep.acc_clamped or rep.pos_clamped):
            self.core.qdot_prev = rep.dq / dt
        # RAIL_ONLY: arm was told to freeze (twist~0 and qdot_ff arm portion
        # empty), so we still enforce the plan's rail position exactly (the
        # arm QP has no legitimate reason to move the rail in that mode).
        # TCP_FIXED and COUPLED: DO NOT override — safety.clamp's rail a_max
        # limit only takes effect when the integrator is allowed to observe
        # the QP output.  The rail-vel pin in the QP box already forces the
        # QP to output v_rail; safety.clamp then applies a_max_rail_m_s2 as a
        # second (protective) rate limit.
        if rail_only and qdot_ff is not None and dt > 1e-9:
            v_rail = float(np.asarray(qdot_ff)[0])
            self.q_cmd[0] = q_prev[0] + v_rail * dt
            self.q_cmd[1:] = q_prev[1:]
            self.core.qdot_prev[1:] = 0.0
            self.core.qdot_prev[0] = v_rail
        else:
            self._pin_rail_if_locked_hold()
        qdot_out = r.qdot.copy()
        if locked_hold and self.cfg.rail.lock_hard_pin:
            qdot_out[0] = 0.0
        elif plan_drives_rail and qdot_ff is not None:
            qdot_out[0] = float(np.asarray(qdot_ff)[0])
            if rail_only:
                qdot_out[1:] = 0.0
        self.last_sigma_min = r.sigma_min
        return JointIkStep(
            q_send=self.q_cmd.copy(),
            qdot=qdot_out,
            twist_base=twist_base,
            sigma_min=r.sigma_min,
            manip=r.manip,
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/loop.py` L1144–L1161

*loop.py — _rail_m_for_init / _rail_m_for_feedback*

```python
def _rail_m_for_init(rail_bridge, inner: JointIkController) -> float:
    """Seed WBC ``q_cmd[0]`` at task/phase start.

    Use the open-loop command stream (what the motor was told), not the encoder.
    Encoder feedback in the WBC froze the governor, blocked pose-D arrival, and
    made force-hybrid unreachable; it also stacked incremental segments to travel.
    """
    if rail_bridge is not None and rail_bridge.enabled:
        return float(rail_bridge.commanded_m)
    return float(inner.q_cmd[0])


def _rail_m_for_feedback(_rail_bridge, inner: JointIkController) -> float:
    """Rail component of ``q_meas`` inside the WBC tick: always ``q_cmd[0]``.

    Motor tracking is open-loop in ``RailServoBridge``; encoder is SHM/twin only.
    """
    return float(inner.q_cmd[0])
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/loop.py` L1308–L1320

*loop.py — phase start q0 rail seed*

```python
        _pose0_rm = async_obs.wait_first_pose(timeout_s=5.0)
        snap0 = async_obs.read()
        if snap0.q_deg is None:
            raise RuntimeError("no joint feedback from robot")
        q0_rad = _expand_q_meas(
            deg2rad(snap0.q_deg),
            _rail_m_for_init(rail_bridge, inner),
        )
        # The whole Cartesian loop (inner and outer) uses the Pinocchio tcp
        # frame; Realman FK for the active tool may differ.
        pose0 = inner.kin.fk_pose(q0_rad)
        inner.reset(q0_rad)
        if snap0.pose is not None:
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/loop.py` L1366–L1372

*loop.py — phase enter q_meas rail*

```python
                    # Phase origin from the ENCODERS, never from the command integrator.
                    snap = async_obs.read()
                    if snap.q_deg is not None:
                        q_meas = _expand_q_meas(
                            deg2rad(snap.q_deg),
                            _rail_m_for_feedback(rail_bridge, inner),
                        )
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/loop.py` L1412–L1420

*loop.py — tick q_meas rail*

```python
                        snap = async_obs.read()
                        if snap.pose is not None:
                            pose_rm = snap.pose
                        if snap.q_deg is not None:
                            q_meas = _expand_q_meas(
                            deg2rad(snap.q_deg),
                            _rail_m_for_feedback(rail_bridge, inner),
                        )
                            pose_pin = inner.kin.fk_pose(q_meas)
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/loop.py` L1477–L1486

*loop.py — 每 tick rail_bridge.set_target_m(q_cmd[0])*

```python
                        step = inner.update(
                            twist,
                            dt,
                            q_meas=q_meas,
                            qdot_ff=qdot_ff,
                            vel_ff=vel_ff_ref,
                        )
                        if rail_bridge is not None:
                            rail_bridge.set_target_m(float(inner.q_cmd[0]))
                        outer_err_mm = getattr(phase.outer, "last_err_mm", None)
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/loop.py` L1544–L1565

*loop.py — scan debug log rail cmd/meas*

```python
                        # Scan-phase debug log (throttled ~1 Hz): tool-Y sweep, rail, force.
                        if _is_scan and (t_wall - _scan_log_t) >= 1.0:
                            _scan_log_t = t_wall
                            if _scan_origin_pose is None:
                                _scan_origin_pose = pose_pin.copy()
                            dy_cmd_mm = float((pose_cmd[1] - _scan_origin_pose[1]) * 1000.0)
                            dy_meas_mm = float((pose_pin[1] - _scan_origin_pose[1]) * 1000.0)
                            rail_cmd_mm = float(inner.q_cmd[0] * 1000.0)
                            rail_meas_mm = (
                                float(rail_bridge.measured_m * 1000.0)
                                if rail_bridge is not None and rail_bridge.enabled
                                else rail_cmd_mm
                            )
                            fz = float(f_ext[2])
                            print(
                                f"  [scan t={t_ref:5.1f}s] toolY cmd={dy_cmd_mm:+7.1f} "
                                f"meas={dy_meas_mm:+7.1f} mm | rail cmd={rail_cmd_mm:6.1f} "
                                f"meas={rail_meas_mm:6.1f} mm | Fz={fz:+5.2f}N "
                                f"| track={step.cart_err_mm:5.1f}mm gov={scale:.2f} "
                                f"σ={step.sigma_min:.3f}",
                                flush=True,
                            )
```

### `rm75_control/rm75_control/control/admittance_common/phase_ipc.py` L165–L170

*phase_ipc.py — request_stop (Ctrl-C 停 task)*

```python
    def should_stop(self) -> bool:
        return int(self._ctl["stop_req"]) != 0

    def request_stop(self) -> None:
        """Ask the running task to exit (Ctrl-C / emergency)."""
        self._ctl["stop_req"] = np.uint8(1)
```

