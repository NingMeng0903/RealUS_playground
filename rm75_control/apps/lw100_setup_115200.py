#!/usr/bin/env python3
"""Configure USR-TCP232-304 serial via web CGI, then raise LW100 Modbus baud to 115200.

Final target (both sides): **115200 8N1**.

LW100 factory is 9600 8N2, so we must:
  1) set USR -> 9600 8N2 (match factory)
  2) Modbus write FA72=1152, FA73=3
  3) power-cycle LW100 (user must do this)
  4) set USR -> 115200 8N1

Usage::

  source env.sh
  python apps/lw100_setup_115200.py
  # power-cycle LW100, then:
  python apps/lw100_setup_115200.py --verify-only
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from base64 import b64encode

from rm75_control.hw.lw100.drive import (
    FA72_BAUD_115200,
    FA73_PROTO_8N1,
    LW100Drive,
    LW100DriveConfig,
)
from rm75_control.hw.lw100.modbus_rtu_tcp import ModbusRtuTcpClient, ModbusRtuTcpConfig
from rm75_control.hw.lw100.registers import probe_register_map


DEFAULT_HOST = "192.168.0.7"
DEFAULT_PORT = 8234


class UsrTcp232Web:
    """Minimal HTTP client for USR-TCP232-304 web config (admin/admin)."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        *,
        user: str = "admin",
        password: str = "admin",
        timeout_s: float = 5.0,
    ) -> None:
        self.host = host
        self.timeout_s = timeout_s
        token = b64encode(f"{user}:{password}".encode()).decode()
        self._auth = f"Basic {token}"

    def _get(self, path: str) -> str:
        url = f"http://{self.host}{path}"
        req = urllib.request.Request(url, headers={"Authorization": self._auth})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return resp.read().decode("utf-8", errors="replace")

    def read_port_vars(self) -> dict[str, str]:
        html = self._get("/port.shtml")
        # Embedded: var _br = 9600; var sb = 2; ...
        keys = {
            "_br": "br",
            "sb": "stop",
            "bc": "bc",
            "par": "parity",
            "_tlp": "tlp",
            "_trp": "trp",
            "tnm": "tnm",
            "rip": "rip",
            "cmode": "cmode",
            "_cnum": "cnum",
            "umode": "umode",
            "shortc": "shortc",
            "_shortct": "shortct",
            "htpch": "htpch",
            "_htpot": "htpot",
            "htpcoh": "htpcoh",
            "mult": "mult",
        }
        out: dict[str, str] = {}
        for js_name, field in keys.items():
            # match: var _br = 9600;  or var rip = '192.168.0.201';
            import re

            m = re.search(rf"var\s+{js_name}\s*=\s*('?\d[\d.]*'?|'[^']*'|\"[^\"]*\")", html)
            if not m:
                continue
            val = m.group(1).strip().strip("'\"")
            out[field] = val
        return out

    def set_serial(
        self,
        *,
        baud: int,
        data_bits: int = 8,
        parity: int = 0,
        stop_bits: int = 1,
        local_port: int = DEFAULT_PORT,
        work_mode: int = 1,  # 1 = TCP Server
        remote_ip: str = "192.168.0.201",
        remote_port: int = DEFAULT_PORT,
    ) -> None:
        """Save port parameters (does not reboot)."""
        cur = self.read_port_vars()
        params = {
            "port": "0",
            "br": str(baud),
            "bc": str(data_bits),
            "parity": str(parity),
            "stop": str(stop_bits),
            "tlp": str(local_port),
            "trp": str(remote_port),
            "tnm": str(work_mode),
            "mult": cur.get("mult", "0"),
            "rip": remote_ip or cur.get("rip", "192.168.0.201"),
            "umode": cur.get("umode", "0"),
            "shortc": cur.get("shortc", "0"),
            "shortct": cur.get("shortct", "3"),
            "cmode": cur.get("cmode", "0"),
            "cnum": cur.get("cnum", "4"),
            "htpch": cur.get("htpch", "0"),
            "htpurl": "/1.php?",
            "htphead": "User_Agent: Mozilla/4.0\nConnection: close",
            "htpot": cur.get("htpot", "10"),
            "htpcoh": cur.get("htpcoh", "0"),
        }
        qs = urllib.parse.urlencode(params)
        self._get(f"/port.cgi?{qs}")

    def reboot(self) -> None:
        try:
            self._get("/manage.cgi?reset=1&rup=0&rfp=0")
        except (urllib.error.URLError, TimeoutError, OSError):
            # reboot often drops the connection
            pass

    def wait_up(self, timeout_s: float = 30.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                self.read_port_vars()
                return
            except Exception:
                time.sleep(1.0)
        raise TimeoutError(f"USR {self.host} did not come back within {timeout_s:.0f}s")


def _probe_ok(host: str, port: int, slave: int, timeout_s: float = 2.0) -> bool:
    cfg = ModbusRtuTcpConfig(host=host, port=port, slave_id=slave, timeout_s=timeout_s, retries=1)
    try:
        with ModbusRtuTcpClient(cfg) as client:
            probe_register_map(client, expected_slave_id=slave, verbose=True)
        return True
    except Exception as exc:
        print(f"  probe fail: {exc}", flush=True)
        return False


def _write_lw100_115200(host: str, port: int, slave: int) -> None:
    cfg = LW100DriveConfig(
        host=host,
        port=port,
        slave_id=slave,
        timeout_s=2.0,
        verbose=True,
    )
    with LW100Drive(cfg) as drive:
        for line in drive.setup_modbus_serial(
            fa72_baud_code=FA72_BAUD_115200,
            fa73_proto=FA73_PROTO_8N1,
        ):
            print(f"  {line}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--slave", type=int, default=1)
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Only probe at 115200 (after LW100 power-cycle + USR 115200)",
    )
    p.add_argument("--skip-reboot", action="store_true", help="Do not reboot USR (params may not apply)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    usr = UsrTcp232Web(args.host)

    if args.verify_only:
        print(f"verify @ 115200 on {args.host}:{args.port}", flush=True)
        try:
            vars_ = usr.read_port_vars()
            print(f"  USR: br={vars_.get('br')} stop={vars_.get('stop')} "
                  f"parity={vars_.get('parity')} tnm={vars_.get('tnm')} "
                  f"tlp={vars_.get('tlp')}", flush=True)
        except Exception as exc:
            print(f"  WARN: cannot read USR web: {exc}", flush=True)
        if _probe_ok(args.host, args.port, args.slave):
            print("OK: LW100 answers at 115200 8N1", flush=True)
            return 0
        print("FAIL: no Modbus reply at 115200 — power-cycle LW100 and confirm USR is 115200 8N1", flush=True)
        return 1

    print("=== step 1: USR -> 9600 8N2 (match LW100 factory) ===", flush=True)
    try:
        before = usr.read_port_vars()
        print(f"  USR before: br={before.get('br')} stop={before.get('stop')} "
              f"parity={before.get('parity')} tnm={before.get('tnm')} tlp={before.get('tlp')}", flush=True)
        usr.set_serial(baud=9600, data_bits=8, parity=0, stop_bits=2, local_port=args.port, work_mode=1)
        if not args.skip_reboot:
            print("  reboot USR ...", flush=True)
            usr.reboot()
            time.sleep(3.0)
            usr.wait_up(40.0)
        after = usr.read_port_vars()
        print(f"  USR after:  br={after.get('br')} stop={after.get('stop')} "
              f"parity={after.get('parity')} tnm={after.get('tnm')} tlp={after.get('tlp')}", flush=True)
        if after.get("br") != "9600" or after.get("stop") != "2":
            print("FAIL: USR did not apply 9600 8N2", flush=True)
            return 1
    except Exception as exc:
        print(f"FAIL: USR web config: {exc}", flush=True)
        return 1

    print("=== step 2: Modbus probe at 9600 ===", flush=True)
    if not _probe_ok(args.host, args.port, args.slave):
        print(
            "FAIL: LW100 silent at 9600 8N2.\n"
            "  Check RS485: USR A+ -> CN3/CN4 pin5 (RS485+), B- -> pin4 (RS485-), GND -> pin7.\n"
            "  Try swapping A/B. Drive must be powered.",
            flush=True,
        )
        return 1

    print("=== step 3: write FA72=1152 FA73=3 ===", flush=True)
    try:
        _write_lw100_115200(args.host, args.port, args.slave)
    except Exception as exc:
        print(f"FAIL: cannot write FA72/FA73: {exc}", flush=True)
        return 1

    print("=== step 4: USR -> 115200 8N1 ===", flush=True)
    try:
        usr.set_serial(baud=115200, data_bits=8, parity=0, stop_bits=1, local_port=args.port, work_mode=1)
        if not args.skip_reboot:
            print("  reboot USR ...", flush=True)
            usr.reboot()
            time.sleep(3.0)
            usr.wait_up(40.0)
        after = usr.read_port_vars()
        print(f"  USR after:  br={after.get('br')} stop={after.get('stop')} "
              f"parity={after.get('parity')}", flush=True)
    except Exception as exc:
        print(f"FAIL: USR web config: {exc}", flush=True)
        return 1

    print(
        "\n=== DONE (almost) ===\n"
        "LW100 FA72/FA73 were written, but baud change needs a **power-cycle of the servo**.\n"
        "  1) Power off LW100, wait 3s, power on\n"
        "  2) python apps/lw100_setup_115200.py --verify-only\n"
        "  3) python apps/lw100_rail_demo.py --run --move-mm 5 --return -v\n",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
