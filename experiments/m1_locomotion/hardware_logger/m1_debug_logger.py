#!/usr/bin/env python3
"""Keep a bounded, robot-local copy of controller logs and service diagnostics."""
from argparse import ArgumentParser
from collections import Counter
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import subprocess
import time


MAX_FILE_BYTES = 8 * 1024 * 1024
BACKUPS = 7
CHECKPOINT_SECONDS = 5
HEALTH_SECONDS = 10
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SERVICE_PROPERTIES = (
    "ActiveState", "SubState", "MainPID", "Result", "NRestarts", "WorkingDirectory",
)
BENIGN_DDS_MESSAGES = (
    "on_data_available called, samples.length()=",
    "Sample valid=1",
    "Valid sample received, calling callback",
    "Calling OnDataAvailable callback",
)


class RotatingLog:
    def __init__(self, directory):
        self.path = directory / "debug.jsonl"
        self.open()

    def open(self):
        self.stream = self.path.open("ab+")
        self.size = self.stream.tell()
        # Keep the next record readable after a sudden power loss mid-write.
        if self.size:
            self.stream.seek(-1, os.SEEK_END)
            if self.stream.read(1) != b"\n":
                self.stream.write(b"\n")
                self.size += 1

    def sync(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def rotate(self):
        self.sync()
        self.stream.close()
        for index in range(BACKUPS, 0, -1):
            before = self.path if index == 1 else self.path.with_suffix(f".jsonl.{index - 1}")
            after = self.path.with_suffix(f".jsonl.{index}")
            try:
                before.replace(after)
            except FileNotFoundError:
                continue
        self.open()

    def write(self, record):
        line = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        if self.size + len(line) > MAX_FILE_BYTES:
            self.rotate()
        self.stream.write(line)
        self.size += len(line)


def read_cursor(path, boot_id):
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if isinstance(state, dict) and state.get("boot_id") == boot_id:
        cursor = state.get("cursor")
        if isinstance(cursor, str) and cursor:
            return cursor
    return None


def store_cursor(path, boot_id, cursor):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump({"boot_id": boot_id, "cursor": cursor}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def now(boot_id, kind):
    return {"type": kind, "boot_id": boot_id,
            "monotonic_us": time.monotonic_ns() // 1000,
            "realtime_us": time.time_ns() // 1000}


def service_health(directory, unit, boot_id):
    record = now(boot_id, "health")
    record["unit"] = unit
    record["log_disk_free_bytes"] = shutil.disk_usage(directory).free
    record["load_1m"] = os.getloadavg()[0]
    try:
        state = subprocess.run(
            ["systemctl", "show", "--no-pager", unit,
             *(f"--property={key}" for key in SERVICE_PROPERTIES)],
            check=True, capture_output=True, text=True, timeout=3,
        )
        record["service"] = dict(line.split("=", 1) for line in state.stdout.splitlines() if "=" in line)
        pid = int(record["service"].get("MainPID", 0))
        if pid > 0:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                name, _, value = line.partition(":")
                if name == "VmRSS":
                    record["controller_rss_kib"] = int(value.split()[0])
                elif name == "Threads":
                    record["controller_threads"] = int(value)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        record["health_error"] = str(exc)
    return record


def journal_message(entry):
    message = entry.get("MESSAGE")
    if isinstance(message, list):
        message = bytes(message).decode("utf-8", errors="replace")
    if not isinstance(message, str):
        return ""
    return ANSI_ESCAPE.sub("", message).strip()[:4096]


def journal_int(entry, key):
    try:
        return int(entry[key])
    except (KeyError, TypeError, ValueError):
        return None


def follow(directory, unit):
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip().replace("-", "")
    cursor_path = directory / ".journal-cursor.json"
    original_cursor = cursor = read_cursor(cursor_path, boot_id)
    log = RotatingLog(directory)
    log.write({**now(boot_id, "logger_start"), "unit": unit, "resumed": bool(cursor)})
    counts = Counter()
    saw_motor_alive = False
    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    def checkpoint():
        if counts:
            log.write({**now(boot_id, "trace_counts"), "counts": dict(counts)})
            counts.clear()
        log.sync()
        store_cursor(cursor_path, boot_id, cursor)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    args = ["journalctl", "--boot", f"--unit={unit}", "--follow", "--lines=all", "--no-pager",
            "--quiet", "--output=json", "--output-fields=MESSAGE,PRIORITY,_PID,SYSLOG_IDENTIFIER"]
    if cursor:
        args.append(f"--after-cursor={cursor}")
    child = subprocess.Popen(args, stdout=subprocess.PIPE, bufsize=0)
    try:
        buffer = b""
        next_health = time.monotonic()
        next_checkpoint = next_health + CHECKPOINT_SECONDS
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout, selectors.EVENT_READ)
            while not stopped:
                deadline = min(next_health, next_checkpoint)
                if selector.select(timeout=max(0, deadline - time.monotonic())):
                    chunk = os.read(child.stdout.fileno(), 65536)
                    if not chunk:
                        if stopped:
                            break
                        if cursor == original_cursor:
                            cursor = None
                        raise RuntimeError("The controller journal reader exited")
                    *lines, buffer = (buffer + chunk).split(b"\n")
                    for line in lines:
                        try:
                            entry = json.loads(line)
                            message = journal_message(entry)
                        except (AttributeError, TypeError, ValueError):
                            counts["invalid_journal_record"] += 1
                            continue
                        cursor = entry.get("__CURSOR", cursor)
                        if message.startswith("[DdsReaderListener]") and any(
                            part in message for part in BENIGN_DDS_MESSAGES
                        ):
                            counts["dds_reader_trace"] += 1
                        elif "[INFO M] Motor is alive. Running robot..." in message and saw_motor_alive:
                            counts["motor_heartbeat"] += 1
                        elif message:
                            saw_motor_alive |= "[INFO M] Motor is alive. Running robot..." in message
                            log.write({"type": "controller", "unit": unit,
                                       "boot_id": entry.get("_BOOT_ID", boot_id),
                                       "monotonic_us": journal_int(entry, "__MONOTONIC_TIMESTAMP"),
                                       "realtime_us": journal_int(entry, "__REALTIME_TIMESTAMP"),
                                       "cursor": cursor, "priority": journal_int(entry, "PRIORITY"),
                                       "pid": journal_int(entry, "_PID"), "message": message})
                t = time.monotonic()
                if t >= next_health and not stopped:
                    log.write(service_health(directory, unit, boot_id))
                    next_health = t + HEALTH_SECONDS
                if t >= next_checkpoint:
                    checkpoint()
                    next_checkpoint = t + CHECKPOINT_SECONDS
    finally:
        if child.poll() is None:
            child.terminate()
        try:
            child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        child.stdout.close()
        checkpoint()
        log.stream.close()


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--log-directory", type=Path, default=Path("/var/log/dogmimic-m1"))
    parser.add_argument("--unit", default="dg_fsm.service")
    args = parser.parse_args()
    args.log_directory.mkdir(parents=True, exist_ok=True)
    follow(args.log_directory, args.unit)
