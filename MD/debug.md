# LW100 导轨伺服 — 第三方调试文档

> 项目路径：`/media/camp/EXT_DRIVE/RealUS_playground/rm75_control`  
> 驱动代码：`rm75_control/hw/lw100/`  
> 最后更新：2026-07-22

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
