"""Xbox pad transport + axis layout.

Logical order (what ``gamepad_twist`` consumes) is always:

    [lx, ly, lt, rx, ry, rt]

``lt`` / ``rt`` are left in the Linux wired convention: rest = −1, pressed = +1.
Layouts that already report 0..1 are converted back to that convention so
``normalize_trigger`` stays one function.

Device pick: USB/wired wins over Bluetooth when both are present.
Layout pick: rest-pose classifier, optional JSON pin from identify_gamepad.py.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

LOGICAL_AXES = ("lx", "ly", "lt", "rx", "ry", "rt")
LOGICAL_BUTTONS = ("a", "b", "x", "y", "lb", "rb", "l3", "r3")

# SDL / xpad (USB): LT=2 RT=5, rest −1.
LAYOUT_WIRED_XPAD = "wired_xpad"
# xpadneo-style BT: LX LY RX RY RT LT (this Series X: axis4=RT, axis5=LT).
LAYOUT_BT_XPADNEO = "bt_xpadneo"
# Some BT stacks rest every axis at 0; triggers are already 0..1.
LAYOUT_BT_REST0 = "bt_rest0"

_WIRELESS_NAME = re.compile(r"wireless|bluetooth|xpadneo|series\s*x|xbox series", re.I)
_WIRED_NAME = re.compile(r"x-box 360 pad|\bwired\b|xpad(?!neo)", re.I)
_BT_MAC = re.compile(r"^[0-9a-f]{2}(:[0-9a-f]{2}){5}\b", re.I)
_JS_HANDLER = re.compile(r"\bjs\d+\b", re.I)
_PAD_NAME = re.compile(r"controller|gamepad|joystick|x-box|xbox|xpad", re.I)

LINUX_INPUT_DEVICES = Path("/proc/bus/input/devices")
BUS_USB = 0x0003
BUS_BLUETOOTH = 0x0005

DEFAULT_LAYOUT_PATH = Path("var/gamepad_layout.json")


@dataclass(frozen=True)
class PadLayout:
    """Physical index → logical Xbox control."""

    name: str
    axis_index: dict[str, int] = field(
        default_factory=lambda: {
            "lx": 0,
            "ly": 1,
            "lt": 2,
            "rx": 3,
            "ry": 4,
            "rt": 5,
        }
    )
    # +1 keeps the raw sign; −1 flips (some stacks invert a stick).
    axis_sign: dict[str, int] = field(
        default_factory=lambda: {k: 1 for k in LOGICAL_AXES}
    )
    # −1: Linux xpad trigger rest.  0: trigger already in [0, 1].
    trigger_rest: float = -1.0
    # +1: LB=+Z LT=−Z.  −1 swaps them (this Series X).
    z_sign: int = 1
    button_index: dict[str, int] = field(
        default_factory=lambda: {
            "a": 0,
            "b": 1,
            "x": 2,
            "y": 3,
            "lb": 4,
            "rb": 5,
            "l3": 13,
            "r3": 14,
        }
    )

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, raw: dict) -> PadLayout:
        name = str(raw.get("name") or LAYOUT_WIRED_XPAD)
        if name == LAYOUT_BT_XPADNEO:
            base = layout_bt_xpadneo()
        elif name == LAYOUT_BT_REST0:
            base = layout_bt_rest0()
        else:
            base = layout_wired_xpad()
        axis_index = dict(base.axis_index)
        axis_index.update({str(k): int(v) for k, v in dict(raw.get("axis_index") or {}).items()})
        axis_sign = dict(base.axis_sign)
        axis_sign.update({str(k): int(v) for k, v in dict(raw.get("axis_sign") or {}).items()})
        button_index = dict(base.button_index)
        button_index.update(
            {str(k): int(v) for k, v in dict(raw.get("button_index") or {}).items()}
        )
        return cls(
            name=name,
            axis_index=axis_index,
            axis_sign=axis_sign,
            trigger_rest=float(raw.get("trigger_rest", base.trigger_rest)),
            z_sign=int(raw.get("z_sign", base.z_sign) or 1),
            button_index=button_index,
        )


def layout_wired_xpad() -> PadLayout:
    return PadLayout(name=LAYOUT_WIRED_XPAD)


def layout_bt_xpadneo() -> PadLayout:
    return PadLayout(
        name=LAYOUT_BT_XPADNEO,
        axis_index={"lx": 0, "ly": 1, "lt": 5, "rx": 2, "ry": 3, "rt": 4},
        # This Series X / pygame reports stick-up as LY=+1 (SDL is −1).
        # Flip so gamepad_twist "up → world +X" matches the hand.
        axis_sign={"lx": 1, "ly": -1, "lt": 1, "rx": 1, "ry": 1, "rt": 1},
        z_sign=-1,
        # xpadneo: A B X Y View Menu LB RB …  (LB is 6, not wired 4)
        button_index={"a": 0, "b": 1, "x": 2, "y": 3, "lb": 6, "rb": 7, "l3": 13, "r3": 14},
    )


def layout_bt_rest0() -> PadLayout:
    return PadLayout(name=LAYOUT_BT_REST0, trigger_rest=0.0)


def transport_from_name(name: str) -> str:
    text = str(name or "")
    if _WIRELESS_NAME.search(text):
        return "bluetooth"
    if _WIRED_NAME.search(text):
        return "usb"
    return "unknown"


def transport_from_bus(bus: int | None) -> str:
    if bus is None:
        return "unknown"
    code = int(bus)
    if code == BUS_BLUETOOTH:
        return "bluetooth"
    if code == BUS_USB:
        return "usb"
    return "unknown"


def transport_from_guid(guid: str) -> str:
    """SDL2 joystick GUID: first uint16 is the kernel bus, little-endian hex."""

    text = str(guid or "").strip().lower()
    if len(text) < 4 or any(ch not in "0123456789abcdef" for ch in text[:4]):
        return "unknown"
    bus = int(text[0:2], 16) | (int(text[2:4], 16) << 8)
    return transport_from_bus(bus)


def transport_from_phys(phys: str) -> str:
    text = str(phys or "").strip()
    if text.lower().startswith("usb"):
        return "usb"
    if _BT_MAC.match(text):
        return "bluetooth"
    return "unknown"


def parse_linux_input_devices(text: str) -> list[dict]:
    """Parse ``/proc/bus/input/devices`` into joystick-like records."""

    devices: list[dict] = []
    cur: dict = {}

    def _flush() -> None:
        if not cur:
            return
        handlers = str(cur.get("handlers") or "")
        name = str(cur.get("name") or "")
        if _JS_HANDLER.search(handlers) or _PAD_NAME.search(name):
            devices.append(dict(cur))

    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line:
            _flush()
            cur = {}
            continue
        if line.startswith("I:"):
            bus_m = re.search(r"Bus=([0-9a-fA-F]+)", line)
            cur["bus"] = int(bus_m.group(1), 16) if bus_m else None
        elif line.startswith("N: Name="):
            cur["name"] = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("P: Phys="):
            cur["phys"] = line.split("=", 1)[1].strip()
        elif line.startswith("H: Handlers="):
            cur["handlers"] = line.split("=", 1)[1].strip()
    _flush()
    return devices


def linux_input_devices(path: str | Path | None = None) -> list[dict]:
    p = Path(path) if path is not None else LINUX_INPUT_DEVICES
    try:
        return parse_linux_input_devices(p.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return []


def classify_link_transport(
    *,
    name: str = "",
    guid: str = "",
    bus: int | None = None,
    phys: str = "",
    linux_devices: list[dict] | None = None,
) -> str:
    """USB vs Bluetooth link. GUID/bus/phys win; pygame vs kernel names differ."""

    from_bus = transport_from_bus(bus)
    if from_bus != "unknown":
        return from_bus
    from_guid = transport_from_guid(guid)
    if from_guid != "unknown":
        return from_guid
    from_phys = transport_from_phys(phys)
    if from_phys != "unknown":
        return from_phys
    matches = []
    for dev in linux_devices if linux_devices is not None else linux_input_devices():
        kind = classify_link_transport(
            bus=dev.get("bus"),
            phys=str(dev.get("phys") or ""),
            linux_devices=[],
        )
        if kind != "unknown":
            matches.append(kind)
    unique = tuple(dict.fromkeys(matches))
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        return "unknown"
    return transport_from_name(name)


def bluetooth_joystick_present(linux_devices: list[dict] | None = None) -> bool:
    """True if the kernel currently has a Bluetooth joystick.

    Do not match pygame names.  This bench reports kernel
    ``Xbox Wireless Controller`` and SDL ``Xbox Series X Controller``.
    """

    devices = linux_devices if linux_devices is not None else linux_input_devices()
    for dev in devices:
        kind = classify_link_transport(
            bus=dev.get("bus"),
            phys=str(dev.get("phys") or ""),
            linux_devices=[],
        )
        if kind != "bluetooth":
            continue
        handlers = str(dev.get("handlers") or "")
        if _JS_HANDLER.search(handlers) or _PAD_NAME.search(str(dev.get("name") or "")):
            return True
    return False


def bluetooth_link_live(
    *,
    guid: str = "",
    linux_devices: list[dict] | None = None,
) -> bool:
    """Live Bluetooth pad only. USB-only and missing are false.

    Kernel table wins over a stale pygame object.  A GUID of 0500 with an
    empty kernel list is a dropped link, not a live pad.
    """

    if linux_devices is not None or LINUX_INPUT_DEVICES.is_file():
        return bluetooth_joystick_present(linux_devices)
    return transport_from_guid(guid) == "bluetooth"


def device_priority(name: str, *, transport: str | None = None) -> int:
    """Higher wins. USB/wired always outranks Bluetooth."""

    kind = transport or transport_from_name(name)
    if kind == "usb":
        return 30
    if kind == "bluetooth":
        return 10
    # Unknown: prefer names that do not look wireless.
    if _WIRELESS_NAME.search(str(name or "")):
        return 10
    return 20


def pick_device_index(
    names: list[str],
    *,
    transports: list[str] | None = None,
    require_transport: str | None = None,
) -> int:
    if not names:
        raise ValueError("no joysticks")
    kinds = (
        [str(t) for t in transports]
        if transports is not None
        else [transport_from_name(n) for n in names]
    )
    if len(kinds) < len(names):
        kinds.extend("unknown" for _ in range(len(names) - len(kinds)))
    candidates = list(range(len(names)))
    want = str(require_transport or "").strip()
    if want:
        candidates = [i for i in candidates if kinds[i] == want]
        if not candidates:
            raise ValueError(f"no {want} joystick")
    ranked = sorted(
        candidates,
        key=lambda i: (-device_priority(names[i], transport=kinds[i]), i),
    )
    return int(ranked[0])


def classify_layout(
    axes: np.ndarray,
    *,
    name: str = "",
    rest_eps: float = 0.20,
) -> PadLayout:
    """Guess layout from a no-touch sample.

    Wired xpad:     [lx, ly, lt≈−1, rx, ry, rt≈−1]
    BT xpadneo:     [lx, ly, rx≈0, ry≈0, rt≈−1, lt≈−1]
    BT rest-0:      all ≈ 0  (triggers live in 0..1)
    """

    raw = np.asarray(axes, dtype=float).reshape(-1)
    a = np.zeros(6, dtype=float)
    a[: min(6, raw.size)] = raw[:6]
    near0 = lambda i: abs(float(a[i])) <= rest_eps
    near_neg1 = lambda i: float(a[i]) <= -1.0 + rest_eps

    kind = transport_from_name(name)
    if near_neg1(2) and near_neg1(5):
        return layout_wired_xpad()
    if near_neg1(4) and near_neg1(5) and near0(2):
        return layout_bt_xpadneo()
    if all(near0(i) for i in range(6)):
        # USB that has not streamed yet can also look like this; name breaks the tie.
        if kind == "usb":
            return layout_wired_xpad()
        return layout_bt_rest0()
    if kind == "bluetooth":
        return layout_bt_xpadneo()
    return layout_wired_xpad()


def _trigger_to_wired(raw: float, trigger_rest: float) -> float:
    v = float(raw)
    if float(trigger_rest) >= -0.25:
        return float(np.clip(2.0 * v - 1.0, -1.0, 1.0))
    return float(np.clip(v, -1.0, 1.0))


def apply_layout(
    axes: np.ndarray,
    buttons: np.ndarray,
    layout: PadLayout,
) -> tuple[np.ndarray, np.ndarray]:
    """Physical pygame sample → logical wired-convention axes / buttons."""

    src_ax = np.asarray(axes, dtype=float).reshape(-1)
    src_btn = np.asarray(buttons, dtype=float).reshape(-1)
    out_ax = np.zeros(6, dtype=float)
    for i, key in enumerate(LOGICAL_AXES):
        idx = int(layout.axis_index.get(key, i))
        sign = int(layout.axis_sign.get(key, 1)) or 1
        val = float(src_ax[idx]) if 0 <= idx < src_ax.size else 0.0
        val *= float(sign)
        if key in ("lt", "rt"):
            val = _trigger_to_wired(val, layout.trigger_rest)
        out_ax[i] = val
    out_btn = np.zeros(16, dtype=float)
    out_btn[: min(16, src_btn.size)] = src_btn[:16]
    for logical_i, key in enumerate(LOGICAL_BUTTONS):
        idx = int(layout.button_index.get(key, logical_i))
        out_btn[logical_i] = (
            1.0 if 0 <= idx < src_btn.size and float(src_btn[idx]) > 0.5 else 0.0
        )
    return out_ax, out_btn


def load_identify_record(path: str | Path | None = None) -> dict | None:
    p = Path(path) if path else DEFAULT_LAYOUT_PATH
    if not p.is_file():
        return None
    raw = json.loads(p.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else None


def load_pinned_layout(
    path: str | Path | None = None,
    *,
    name: str | None = None,
    guid: str | None = None,
) -> PadLayout | None:
    raw = load_identify_record(path)
    if raw is None:
        return None
    if name and raw.get("name") and str(raw["name"]) != str(name):
        return None
    if guid and raw.get("guid") and str(raw["guid"]) != str(guid):
        return None
    layout_raw = raw.get("layout") if "layout" in raw else raw
    if not isinstance(layout_raw, dict):
        return None
    layout = PadLayout.from_json(layout_raw)
    # Incomplete identify dumps used to pin name=wired_xpad with only
    # button_index. Series X on this bench is bluetooth / xpadneo.
    if name and transport_from_name(name) == "bluetooth" and layout.name == LAYOUT_WIRED_XPAD:
        neo = layout_bt_xpadneo()
        buttons = dict(neo.button_index)
        buttons.update(layout.button_index)
        return PadLayout(
            name=LAYOUT_BT_XPADNEO,
            axis_index=dict(neo.axis_index),
            axis_sign=dict(neo.axis_sign),
            trigger_rest=neo.trigger_rest,
            z_sign=int(neo.z_sign),
            button_index=buttons,
        )
    return layout


def save_identify_result(path: str | Path, payload: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p
