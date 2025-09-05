import os
import time
import threading
from functools import cached_property
from typing import Any

from .samples import Samples
import pandas as pd
import psutil


class Sampler:
    def __init__(
            self,
            interval_s: float = 1.,
            include_gpu: bool = True,
    ):
        self.interval_s = interval_s
        self.include_gpu = include_gpu
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._records: list[dict[str, Any]] = []

    # start background sampler
    def __enter__(self) -> "Sampler":
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    # stop and join
    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None

    @cached_property
    def samples(self) -> Samples:
        df = (
            pd.DataFrame(self._records)
            .sort_values("t")
            .reset_index(drop=True)
        )
        result = Samples.from_frame(df)
        return result



    # internal sampling loop
    def _run(self):
        proc = psutil.Process(os.getpid())

        # optional CUDA import guarded
        torch = None
        if self.include_gpu:
            try:
                import torch as _torch  # type: ignore
                torch = _torch
            except Exception:
                torch = None

        t0 = time.monotonic()

        while not self._stop.is_set():
            now = time.monotonic() - t0

            # system memory snapshot
            vm = psutil.virtual_memory()
            sm = psutil.swap_memory()

            # CPU snapshot
            cpu_percent = float(psutil.cpu_percent(interval=None))
            cpu_cores = int(psutil.cpu_count(logical=True) or 0)

            # process tree RSS
            try:
                rss_self = proc.memory_info().rss
            except psutil.Error:
                rss_self = 0

            rss_children = 0
            try:
                for ch in proc.children(recursive=True):
                    try:
                        rss_children += ch.memory_info().rss
                    except psutil.Error:
                        pass
            except psutil.Error:
                pass

            # GPU (optional)
            gpu_used_gb = float("nan")
            gpu_total_gb = float("nan")
            gpu_percent = float("nan")
            gpu_util_percent = float("nan")
            gpu_cores = float("nan")
            if torch is not None and torch.cuda.is_available():
                try:
                    dev = torch.cuda.current_device()
                    free_b, total_b = torch.cuda.mem_get_info(dev)
                    used_b = total_b - free_b
                    gpu_used_gb = used_b / (1024 ** 3)
                    gpu_total_gb = total_b / (1024 ** 3)
                    gpu_percent = (used_b / total_b) * 100 if total_b else float("nan")

                    # try NVML for utilization and cores
                    try:
                        from pynvml import (
                            nvmlInit,
                            nvmlShutdown,
                            nvmlDeviceGetHandleByIndex,
                            nvmlDeviceGetUtilizationRates,
                            nvmlDeviceGetMultiprocessorCount,
                        )
                        nvmlInit()
                        handle = nvmlDeviceGetHandleByIndex(int(dev))
                        util = nvmlDeviceGetUtilizationRates(handle)
                        gpu_util_percent = float(util.gpu)
                        try:
                            gpu_cores = float(nvmlDeviceGetMultiprocessorCount(handle))
                        except Exception:
                            gpu_cores = float("nan")
                        nvmlShutdown()
                    except Exception:
                        # fallback via nvidia-smi, if available
                        try:
                            import subprocess
                            out = subprocess.check_output(
                                [
                                    "nvidia-smi",
                                    "--query-gpu=utilization.gpu",
                                    "--format=csv,noheader,nounits",
                                    f"-i{int(dev)}",
                                ],
                                timeout=0.2,
                            )
                            # may return like b'35\n'
                            gpu_util_percent = float(out.decode().strip().splitlines()[0])
                        except Exception:
                            pass
                except Exception:
                    pass

            rec = {
                "t": now,
                "sys_used_gb": vm.used / (1024 ** 3),
                "sys_avail_gb": vm.available / (1024 ** 3),
                "sys_percent": float(vm.percent),
                "proc_rss_gb": rss_self / (1024 ** 3),
                "children_rss_gb": rss_children / (1024 ** 3),
                "job_rss_gb": (rss_self + rss_children) / (1024 ** 3),
                "swap_used_gb": sm.used / (1024 ** 3),
                "swap_percent": float(sm.percent),
                "cpu_percent": cpu_percent,
                "cpu_cores": cpu_cores,
                "gpu_used_gb": gpu_used_gb,
                "gpu_total_gb": gpu_total_gb,
                "gpu_percent": gpu_percent,
                "gpu_util_percent": gpu_util_percent,
                "gpu_cores": gpu_cores,
            }
            self._records.append(rec)

            # warning when close to limits
            if rec["sys_avail_gb"] < 2.0 or rec["swap_percent"] > 50:
                # keep it silent—tqdm postfix already surfaces pressure;
                # if you prefer logs, hook your logger here
                pass

            self._stop.wait(self.interval_s)
