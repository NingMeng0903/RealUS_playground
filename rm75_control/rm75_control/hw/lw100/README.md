# LW100 rail servo (Modbus RTU over USR-TCP232-304)

## Network layout

| Device | IP | Role |
|--------|-----|------|
| PC | `192.168.1.80` + `192.168.0.80` | RM75 UDP target + Modbus TCP client |
| RM75 | `192.168.1.18` | Arm TCP/UDP |
| USR-TCP232-304 | `192.168.0.7` | RS485 ↔ TCP transparent bridge |

## USR-TCP232-304 (verify before demo)

1. **Work mode**: TCP **Server** (not Client).
2. **Local port**: `8234` (or match `--port` in demo).
3. **Serial** (must match LW100 `FA72` / `FA73`):

| Setting | LW100 **factory default** | USR (you set) | Match? |
|---------|---------------------------|---------------|--------|
| Baud | **9600** (`FA72=96`) | 115200 | often **NO** → timeout |
| Format | **8N2** (`FA73=0`) | 8N1 | often **NO** |

**First-time bring-up (no drive keypad — host configures baud):**

1. USR **9600 / 8 / NONE / 2 stop** (match factory LW100).
2. `python apps/lw100_rail_demo.py --diagnose`  → must read FA71=1.
3. `python apps/lw100_rail_demo.py --setup-serial`  → writes FA72=1152, FA73=3 via Modbus.
4. **Power-cycle LW100**; USR → **115200 8N1**; Save/Restart.
5. `python apps/lw100_rail_demo.py --diagnose` again at high speed.

**Later / faster:** change LW100 panel to `FA72=1152`, `FA73=3` (115200 8N1), save + power-cycle, then USR 115200 8N1.

4. **Protocol**: send **Modbus RTU** frames (with CRC16) over TCP — **not** Modbus-TCP (no MBAP header).

## LW100 control mode (rail)

- `FA4=0` position, `FA14=3` internal position input, `FD-0=0` absolute position.
- Target: `FD-2` (revolutions) + `FD-3` (in-rev pulses) + `FD-4` (segment speed r/min).
- Trigger: internal position P1 via forced DI `FC-18` (POS0–2) + `CTRG` rising edge.
- Enable: `FC-15` bit0 (`SON`) forced ON, or hardwired SON on CN1.

## 1610 ball screw

Standard **1610** → **10 mm/rev** (1:1 motor coupling):

```
travel_mm = revolutions × 10
revolutions = travel_mm / 10
```

Pulses/rev from `FA11` (default 10000).

## Quick test

```bash
cd rm75_control && source env.sh

# One-shot: USR 9600 → write FA72/FA73 → USR 115200
python apps/lw100_setup_115200.py
# Power-cycle LW100, then:
python apps/lw100_setup_115200.py --verify-only

# Motion demo (after verify OK)
python apps/lw100_rail_demo.py --run --move-mm 5 --return -v
```
