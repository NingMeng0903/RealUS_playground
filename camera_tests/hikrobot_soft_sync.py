#!/usr/bin/env python3
"""Test two Hikrobot cameras with paired software triggers using the official SDK.

MVS must release both cameras before running this program. Every modified camera
setting is restored in finally. Shared capture IDs pair frames; they do not claim
simultaneous exposure or synchronized device clocks. Clock mapping is only an
estimate based on timestamp-latch round trips.
"""

import argparse
import concurrent.futures
import csv
import ctypes as ct
import datetime
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from zoneinfo import ZoneInfo

SDK_ROOT = Path(os.environ.get("MVCAM_SDK_PATH", "/opt/MVS"))
os.environ.setdefault("MVCAM_COMMON_RUNENV", str(SDK_ROOT / "lib"))
sys.path.insert(0, str(SDK_ROOT / "Samples/64/Python/MvImport"))
import MvCameraControl_class as sdk


def check(ret, operation):
    if ret:
        raise RuntimeError(f"{operation}: 0x{ret & 0xffffffff:08x}")


def decode(value):
    return bytes(value).split(b"\0", 1)[0].decode("utf-8", errors="replace")


def ipv4(value):
    return ".".join(str((value >> shift) & 255) for shift in (24, 16, 8, 0))


def stats(values):
    if not values:
        return None
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(.95 * (len(ordered) - 1)))],
        "max": ordered[-1],
    }


class Camera:
    def __init__(self, info):
        net = info.SpecialInfo.stGigEInfo
        self.serial = decode(net.chSerialNumber)
        self.ip = ipv4(net.nCurrentIp)
        self.handle = sdk.MvCamera()
        self.created = False
        self.opened = False
        self.grabbing = False
        self.original = {}
        self.last_frame = None
        self.last_ticks = None
        self.clock_samples = {"before": [], "after": []}

        check(self.handle.MV_CC_CreateHandle(info), f"{self.serial} create")
        self.created = True
        try:
            check(self.handle.MV_CC_OpenDevice(sdk.MV_ACCESS_Exclusive, 0),
                  f"{self.serial} open (disconnect MVS first)")
            self.opened = True
        except Exception:
            self.handle.MV_CC_DestroyHandle()
            self.created = False
            raise

    def get(self, kind, key):
        if kind == "enum":
            value = sdk.MVCC_ENUMVALUE()
            ret = self.handle.MV_CC_GetEnumValue(key, value)
            result = value.nCurValue
        elif kind == "int":
            value = sdk.MVCC_INTVALUE_EX()
            ret = self.handle.MV_CC_GetIntValueEx(key, value)
            result = value.nCurValue
        elif kind == "float":
            value = sdk.MVCC_FLOATVALUE()
            ret = self.handle.MV_CC_GetFloatValue(key, value)
            result = value.fCurValue
        else:
            value = ct.c_bool()
            ret = self.handle.MV_CC_GetBoolValue(key, value)
            result = value.value
        check(ret, f"{self.serial} read {key}")
        return result

    def set(self, kind, key, value):
        method = {"enum": self.handle.MV_CC_SetEnumValue,
                  "int": self.handle.MV_CC_SetIntValueEx,
                  "float": self.handle.MV_CC_SetFloatValue,
                  "bool": self.handle.MV_CC_SetBoolValue}[kind]
        check(method(key, value), f"{self.serial} set {key}")

    def snapshot(self):
        required = [
            ("enum", "AcquisitionMode"), ("enum", "TriggerSelector"),
            ("enum", "TriggerMode"), ("enum", "TriggerSource"),
            ("enum", "ExposureAuto"), ("float", "ExposureTime"),
            ("int", "AcquisitionBurstFrameCount"), ("int", "GevSCPSPacketSize"),
            ("int", "GevSCPD"), ("bool", "AutoSCPD"),
            ("bool", "AcquisitionFrameRateEnable"),
        ]
        for kind, key in required:
            self.original[key] = {"kind": kind, "value": self.get(kind, key)}
        for kind, key in [("bool", "TriggerCacheEnable"), ("float", "TriggerDelay")]:
            try:
                self.original[key] = {"kind": kind, "value": self.get(kind, key)}
            except RuntimeError:
                pass
        self.tick_hz = self.get("int", "GevTimestampTickFrequency")
        if not self.tick_hz:
            raise RuntimeError("Invalid timestamp frequency")

    def configure(self, exposure_us, packet_size, scpd):
        check(self.handle.MV_CC_SetEnumValueByString("TriggerMode", "Off"), "disable trigger")
        check(self.handle.MV_CC_SetEnumValueByString("AcquisitionMode", "Continuous"), "continuous mode")
        check(self.handle.MV_CC_SetEnumValueByString("TriggerSelector", "FrameBurstStart"), "trigger selector")
        check(self.handle.MV_CC_SetEnumValueByString("TriggerSource", "Software"), "software trigger")
        self.set("int", "AcquisitionBurstFrameCount", 1)
        self.set("bool", "AcquisitionFrameRateEnable", False)
        self.set("enum", "ExposureAuto", 0)
        self.set("float", "ExposureTime", exposure_us)
        self.set("bool", "AutoSCPD", False)
        self.set("int", "GevSCPSPacketSize", packet_size)
        self.set("int", "GevSCPD", scpd)
        if "TriggerDelay" in self.original:
            self.set("float", "TriggerDelay", 0.)
        if "TriggerCacheEnable" in self.original:
            self.set("bool", "TriggerCacheEnable", False)
        check(self.handle.MV_CC_SetEnumValueByString("TriggerMode", "On"), "enable trigger")

    def calibrate(self, phase, count=16):
        for _ in range(count):
            start = time.perf_counter_ns()
            check(self.handle.MV_CC_SetCommandValue("GevTimestampControlLatch"), "timestamp latch")
            ticks = self.get("int", "GevTimestampValue")
            end = time.perf_counter_ns()
            self.clock_samples[phase].append({
                "device_ticks": ticks,
                "device_ns": ticks * 1_000_000_000 / self.tick_hz,
                "host_mid_ns": (start + end) // 2,
                "rtt_ns": end - start,
            })

    def start(self):
        check(self.handle.MV_CC_StartGrabbing(), f"{self.serial} start")
        self.grabbing = True
        check(self.handle.MV_CC_ClearImageBuffer(), "clear previous buffers")

    def receive(self, timeout_ms, save_path):
        frame = sdk.MV_FRAME_OUT()
        check(self.handle.MV_CC_GetImageBuffer(frame, timeout_ms), f"{self.serial} receive")
        received_ns = time.perf_counter_ns()
        try:
            info = frame.stFrameInfo
            if info.nLostPacket:
                raise RuntimeError(f"{self.serial} incomplete frame: {info.nLostPacket} missing packets")
            frame_no = int(info.nFrameNum)
            ticks = (int(info.nDevTimeStampHigh) << 32) | int(info.nDevTimeStampLow)
            if self.last_frame is not None and ((frame_no - self.last_frame) & 0xffffffff) != 1:
                raise RuntimeError(f"{self.serial} frame discontinuity: {self.last_frame} -> {frame_no}")
            if self.last_ticks is not None and ticks <= self.last_ticks:
                raise RuntimeError(f"{self.serial} device timestamp did not advance")
            self.last_frame, self.last_ticks = frame_no, ticks
            result = {
                "serial": self.serial, "ip": self.ip, "frame_number": frame_no,
                "device_ticks": ticks, "device_tick_hz": self.tick_hz,
                "receive_host_ns": received_ns, "sdk_host_timestamp_raw": int(info.nHostTimeStamp),
                "width": int(info.nWidth), "height": int(info.nHeight),
                "pixel_format": int(info.enPixelType), "data_bytes": int(info.nFrameLen),
                "lost_packets_in_frame": int(info.nLostPacket),
                "device_trigger_index_raw": int(info.nTriggerIndex),
            }
            if not frame.pBufAddr or not info.nFrameLen:
                raise RuntimeError(f"{self.serial} empty frame")
            if save_path:
                # Convert and save a copied frame on the main thread after both
                # receive calls have returned; no shared SDK buffer outlives Free.
                result["image_data"] = ct.string_at(frame.pBufAddr, info.nFrameLen)
                result["sample_path"] = str(save_path)
            return result
        finally:
            check(self.handle.MV_CC_FreeImageBuffer(frame), f"{self.serial} free buffer")

    def save(self, result):
        data = result.pop("image_data")
        buffer = (ct.c_ubyte * len(data)).from_buffer_copy(data)
        path = ct.create_string_buffer(result["sample_path"].encode("ascii"))
        param = sdk.MV_SAVE_IMAGE_TO_FILE_PARAM_EX()
        param.nWidth, param.nHeight = result["width"], result["height"]
        param.enPixelType = result["pixel_format"]
        param.pData = ct.cast(buffer, ct.POINTER(ct.c_ubyte))
        param.nDataLen = len(data)
        param.enImageType = sdk.MV_Image_Bmp
        param.pcImagePath = ct.cast(path, ct.POINTER(ct.c_char))
        param.iMethodValue = 1
        check(self.handle.MV_CC_SaveImageToFileEx(param), f"{self.serial} save BMP")

    def network_stats(self):
        value = sdk.MV_MATCH_INFO_NET_DETECT()
        query = sdk.MV_ALL_MATCH_INFO()
        query.nType = sdk.MV_MATCH_TYPE_NET_DETECT
        query.pInfo = ct.cast(ct.pointer(value), ct.c_void_p)
        query.nInfoSize = ct.sizeof(value)
        check(self.handle.MV_CC_GetAllMatchInfo(query), f"{self.serial} network stats")
        return {name: int(getattr(value, name)) for name, _ in value._fields_}

    def restore_and_close(self):
        errors = []
        if self.grabbing:
            ret = self.handle.MV_CC_StopGrabbing()
            if ret:
                errors.append(f"stop: 0x{ret & 0xffffffff:x}")
            self.grabbing = False
        if self.opened and self.original:
            # Source, selector and exposure writes require acquisition stopped
            # and automatic exposure / triggering disabled.
            self.handle.MV_CC_SetEnumValueByString("TriggerMode", "Off")
            self.handle.MV_CC_SetEnumValueByString("ExposureAuto", "Off")
            restore_order = ["AcquisitionMode", "TriggerSelector", "TriggerSource",
                             "AcquisitionBurstFrameCount", "GevSCPSPacketSize", "GevSCPD", "AutoSCPD",
                             "AcquisitionFrameRateEnable", "TriggerCacheEnable",
                             "TriggerDelay", "ExposureTime", "ExposureAuto", "TriggerMode"]
            for key in restore_order:
                if key in self.original:
                    try:
                        entry = self.original[key]
                        self.set(entry["kind"], key, entry["value"])
                    except Exception as exc:
                        errors.append(str(exc))
            for key, entry in self.original.items():
                try:
                    actual = self.get(entry["kind"], key)
                    if actual != entry["value"]:
                        errors.append(f"{key}: restored {actual}, expected {entry['value']}")
                except Exception as exc:
                    errors.append(str(exc))
        if self.opened:
            ret = self.handle.MV_CC_CloseDevice()
            if ret:
                errors.append(f"close: 0x{ret & 0xffffffff:x}")
            self.opened = False
        if self.created:
            self.handle.MV_CC_DestroyHandle()
            self.created = False
        return errors


def issue_trigger(camera, barrier):
    barrier.wait(timeout=5)
    start = time.perf_counter_ns()
    ret = camera.handle.MV_CC_SetCommandValue("TriggerSoftware")
    end = time.perf_counter_ns()
    return {"serial": camera.serial, "trigger_call_start_ns": start,
            "trigger_call_end_ns": end, "trigger_return_code": ret & 0xffffffff}


def map_device_time(camera, ticks):
    before = min(camera.clock_samples["before"], key=lambda x: x["rtt_ns"])
    after = min(camera.clock_samples["after"], key=lambda x: x["rtt_ns"])
    device_ns = ticks * 1_000_000_000 / camera.tick_hz
    denominator = after["device_ns"] - before["device_ns"]
    if denominator <= 0:
        raise RuntimeError("Clock latch samples did not advance")
    scale = (after["host_mid_ns"] - before["host_mid_ns"]) / denominator
    estimated = before["host_mid_ns"] + scale * (device_ns - before["device_ns"])
    return estimated, max(before["rtt_ns"], after["rtt_ns"]) / 2, scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-ip", default="192.168.0.80")
    parser.add_argument("--serials", nargs=2, default=["DA8689632", "DA8689644"])
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--exposure-us", type=float, default=30000)
    parser.add_argument("--packet-size", type=int, default=8164,
                        help="Requires jumbo packets throughout the network; verified on this setup")
    parser.add_argument("--scpd", type=int, default=10000,
                        help="Packet delay in camera timestamp ticks (10000 = 100 us at 100 MHz)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.pairs < 1 or args.fps <= 0 or len(set(args.serials)) != 2:
        parser.error("Need positive pairs/fps and two different camera serials")
    if args.output is None:
        stamp = datetime.datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        args.output = Path(__file__).resolve().parent / "results" / stamp
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    cameras, rows, rounds = [], [], []
    summary = {"requested_pairs": args.pairs, "requested_fps": args.fps,
               "test_exposure_us": args.exposure_us, "packet_size": args.packet_size,
               "packet_delay_ticks": args.scpd,
               "software_trigger_only": True, "hardware_synchronization_verified": False,
               "output": str(args.output), "errors": [], "restore": {}}
    check(sdk.MvCamera.MV_CC_Initialize(), "initialize SDK")
    try:
        devices = sdk.MV_CC_DEVICE_INFO_LIST()
        check(sdk.MvCamera.MV_CC_EnumDevices(sdk.MV_GIGE_DEVICE, devices), "enumerate cameras")
        infos = {}
        for index in range(devices.nDeviceNum):
            info = ct.cast(devices.pDeviceInfo[index], ct.POINTER(sdk.MV_CC_DEVICE_INFO)).contents
            net = info.SpecialInfo.stGigEInfo
            if ipv4(net.nNetExport) == args.host_ip and decode(net.chSerialNumber) in args.serials:
                infos[decode(net.chSerialNumber)] = info
        if any(serial not in infos for serial in args.serials):
            raise RuntimeError(f"Requested cameras not found on {args.host_ip}: {list(infos)}")
        for serial in args.serials:
            camera = Camera(infos[serial])
            cameras.append(camera)
            camera.snapshot()
        (args.output / "original_settings.json").write_text(json.dumps(
            {c.serial: {"ip": c.ip, "settings": c.original} for c in cameras}, indent=2))
        for camera in cameras:
            camera.configure(args.exposure_us, args.packet_size, args.scpd)
            camera.calibrate("before")
            camera.start()
        # Both armed cameras must remain silent until software trigger.
        for camera in cameras:
            frame = sdk.MV_FRAME_OUT()
            ret = camera.handle.MV_CC_GetImageBuffer(frame, 100)
            if ret == 0:
                camera.handle.MV_CC_FreeImageBuffer(frame)
                raise RuntimeError(f"{camera.serial} produced a frame without trigger")
            if (ret & 0xffffffff) != sdk.MV_E_NODATA:
                check(ret, f"{camera.serial} no-trigger check")
        summary["no_frames_without_trigger"] = True
        print("Both cameras armed; no unsolicited frames. Starting paired triggers.", flush=True)
        beginning = time.perf_counter_ns()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for capture_id in range(1, args.pairs + 1):
                target = beginning + int((capture_id - 1) * 1e9 / args.fps)
                delay = (target - time.perf_counter_ns()) / 1e9
                if delay > 0:
                    time.sleep(delay)
                for camera in cameras:
                    check(camera.handle.MV_CC_ClearImageBuffer(), "clear round buffers")
                save_sample = capture_id in {1, max(1, args.pairs // 2), args.pairs}
                receive_futures = [executor.submit(camera.receive, 1500,
                    args.output / f"pair_{capture_id:04d}_{camera.serial}.bmp" if save_sample else None)
                    for camera in cameras]
                barrier = threading.Barrier(3)
                trigger_futures = [executor.submit(issue_trigger, camera, barrier) for camera in cameras]
                shared_reference = time.perf_counter_ns()
                barrier.wait(timeout=5)
                triggers = [future.result(timeout=5) for future in trigger_futures]
                failures = []
                round_rows = []
                for camera, trigger, future in zip(cameras, triggers, receive_futures):
                    try:
                        result = future.result(timeout=5)
                        result.update(trigger)
                        result.update(capture_id=capture_id, pair_reference_host_ns=shared_reference)
                        if result["trigger_return_code"]:
                            raise RuntimeError(f"{camera.serial} trigger failed: {result['trigger_return_code']:x}")
                        if "image_data" in result:
                            camera.save(result)
                        round_rows.append(result)
                    except Exception as exc:
                        failures.append(str(exc))
                rows.extend(round_rows)
                rounds.append({"capture_id": capture_id, "frames_received": len(round_rows),
                               "errors": failures})
                if failures:
                    # Stop immediately rather than risk pairing a delayed frame
                    # with the next trigger. A new run re-arms and clears queues.
                    raise RuntimeError(f"Round {capture_id} failed: {failures}")
                if capture_id % 20 == 0 or capture_id == args.pairs:
                    print(f"{capture_id}/{args.pairs} complete pairs", flush=True)
        summary["capture_duration_s"] = (time.perf_counter_ns() - beginning) / 1e9
        for camera in cameras:
            camera.calibrate("after")
        summary["network_stats"] = {camera.serial: camera.network_stats() for camera in cameras}
        for row in rows:
            camera = next(c for c in cameras if c.serial == row["serial"])
            estimate, uncertainty, scale = map_device_time(camera, row["device_ticks"])
            row["device_time_on_host_est_ns"] = round(estimate)
            row["clock_anchor_half_rtt_ns"] = round(uncertainty)
            row["clock_scale_estimate"] = scale
    except Exception as exc:
        summary["errors"].append(str(exc))
        print("ERROR:", exc, flush=True)
    finally:
        if "network_stats" not in summary:
            summary["network_stats"] = {}
            for camera in cameras:
                if camera.grabbing:
                    try:
                        summary["network_stats"][camera.serial] = camera.network_stats()
                    except Exception as exc:
                        summary["network_stats"][camera.serial] = {"error": str(exc)}
        for camera in reversed(cameras):
            summary["restore"][camera.serial] = camera.restore_and_close()
        sdk.MvCamera.MV_CC_Finalize()
        summary["complete_pairs"] = sum(r["frames_received"] == 2 and not r["errors"] for r in rounds)
        summary["frame_count"] = len(rows)
        summary["settings_restored"] = all(not e for e in summary["restore"].values())
        if "capture_duration_s" in summary and len(rounds) > 1:
            refs = [row["pair_reference_host_ns"] for row in rows[::2]]
            summary["achieved_trigger_fps"] = (len(refs) - 1) * 1e9 / (refs[-1] - refs[0])
        if rows:
            fieldnames = sorted(set().union(*(row.keys() for row in rows)))
            with (args.output / "frames.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            trigger_gaps, receive_gaps, estimated_device_gaps = [], [], []
            for capture_id in sorted(set(row["capture_id"] for row in rows)):
                pair = [row for row in rows if row["capture_id"] == capture_id]
                if len(pair) != 2:
                    continue
                trigger_gaps.append(abs(pair[0]["trigger_call_start_ns"] - pair[1]["trigger_call_start_ns"]) / 1e6)
                receive_gaps.append(abs(pair[0]["receive_host_ns"] - pair[1]["receive_host_ns"]) / 1e6)
                if all("device_time_on_host_est_ns" in r for r in pair):
                    estimated_device_gaps.append(abs(pair[0]["device_time_on_host_est_ns"] - pair[1]["device_time_on_host_est_ns"]) / 1e6)
            summary["trigger_call_start_gap_ms"] = stats(trigger_gaps)
            summary["receive_return_gap_ms"] = stats(receive_gaps)
            summary["mapped_device_timestamp_gap_est_ms"] = stats(estimated_device_gaps)
            summary["mapped_timestamp_caveat"] = (
                "Estimated from independent camera latch clocks and host round trips; "
                "not a measurement or guarantee of simultaneous exposure. Half-RTT "
                "values describe anchor intervals, not all sources of error.")
        (args.output / "clock_samples.json").write_text(json.dumps(
            {camera.serial: camera.clock_samples for camera in cameras}, indent=2))
        (args.output / "rounds.json").write_text(json.dumps(rounds, indent=2))
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["complete_pairs"] == args.pairs and summary["settings_restored"] and not summary["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
