"""Per-lane plumbing: GPU locks, state files, atomic row appends, telemetry."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import threading
import time
from pathlib import Path

from . import schedule


def cell_dir(run_dir, cell: dict) -> Path:
    return schedule.state_path(run_dir, cell)


def read_state(run_dir, cell: dict) -> dict:
    path = cell_dir(run_dir, cell) / "state.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_state(run_dir, cell: dict, **fields) -> None:
    state = read_state(run_dir, cell)
    state.update(fields)
    state.setdefault("cell_id", cell["id"])
    state["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json_atomic(cell_dir(run_dir, cell) / "state.json", state)


def append_rows(run_dir, cell: dict, rows: list) -> None:
    if not rows:
        return
    path = cell_dir(run_dir, cell) / "rows.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (PermissionError, ProcessLookupError):
        return False


def acquire_locks(run_dir, gpus) -> list:
    """Exclusive per-GPU lock files; stale locks from dead local pids are reused."""
    lock_dir = Path(run_dir) / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    held = []
    try:
        for gpu in gpus:
            path = lock_dir / ("gpu_%s.lock" % gpu)
            while True:
                try:
                    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(fd, str(os.getpid()).encode())
                    os.close(fd)
                    held.append(path)
                    break
                except FileExistsError:
                    try:
                        pid = int(path.read_text(encoding="utf-8").strip())
                    except (OSError, ValueError):
                        pid = 0
                    if pid and _alive(pid):
                        raise RuntimeError("GPU %s is locked by live pid %d" % (gpu, pid))
                    path.unlink(missing_ok=True)
    except Exception:
        release_locks(held)
        raise
    return held


def release_locks(paths) -> None:
    for path in paths or []:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


class Telemetry(threading.Thread):
    """nvidia-smi snapshot every interval seconds until stop()."""

    FIELDS = ("index", "timestamp", "utilization.gpu", "memory.used", "memory.total",
              "temperature.gpu", "power.draw", "clocks.sm")

    def __init__(self, gpus, path: Path, interval: float = 5.0):
        super().__init__(daemon=True)
        self.gpus = list(gpus)
        self.path = Path(path)
        self.interval = float(interval)
        self._stop = threading.Event()

    def run(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.is_file()
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            if new:
                writer.writerow(self.FIELDS)
                handle.flush()
            while not self._stop.is_set():
                try:
                    process = subprocess.run(
                        ["nvidia-smi", "-i", ",".join(str(g) for g in self.gpus),
                         "--query-gpu=" + ",".join(self.FIELDS),
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=4)
                    if process.returncode == 0:
                        for line in process.stdout.strip().splitlines():
                            if line.strip():
                                writer.writerow([part.strip() for part in line.split(",")])
                        handle.flush()
                except Exception:
                    pass
                self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        self.join(timeout=self.interval + 2.0)
