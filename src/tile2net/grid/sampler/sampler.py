import os
import time
import threading
from typing import Any

from .samples import Samples
import pandas as pd
import psutil


class MemSampler:
    def __init__(
            self,
            interval_s: float = 0.5,
            include_gpu: bool = True,
            tqdm_bar=None,
    ):
        self.interval_s = interval_s
        self.include_gpu = include_gpu
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._records: list[dict[str, Any]] = []
        self._bar = tqdm_bar

    # start background sampler
    def __enter__(self) -> "MemSampler":
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

    # def to_samples(
    #         self,
    # ) -> Samples:
    #     df = pd.DataFrame(self._records).sort_values("t").reset_index(drop=True)
    #     return df.pipe(Samples.from_frame)
    #
    # # concise summary stats for benchmarking/regression checks
    # def summary(self) -> dict[str, float]:
    #     df = self.to_frame()
    #     if df.empty:
    #         return {}
    #     q = df.quantile([0.5, 0.95]).to_dict()
    #     out = {
    #         "peak_job_rss_gb": float(df["job_rss_gb"].max()),
    #         "mean_job_rss_gb": float(df["job_rss_gb"].mean()),
    #         "p95_job_rss_gb": float(q["job_rss_gb"][0.95]),
    #         "min_sys_avail_gb": float(df["sys_avail_gb"].min()),
    #         "max_sys_percent": float(df["sys_percent"].max()),
    #         "max_swap_used_gb": float(df["swap_used_gb"].max()),
    #         "max_gpu_used_gb": float(df["gpu_used_gb"].max()) if "gpu_used_gb" in df else float("nan"),
    #     }
    #     return out

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
            if torch is not None and torch.cuda.is_available():
                try:
                    dev = torch.cuda.current_device()
                    free_b, total_b = torch.cuda.mem_get_info(dev)
                    used_b = total_b - free_b
                    gpu_used_gb = used_b / (1024 ** 3)
                    gpu_total_gb = total_b / (1024 ** 3)
                    gpu_percent = (used_b / total_b) * 100 if total_b else float("nan")
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
                "gpu_used_gb": gpu_used_gb,
                "gpu_total_gb": gpu_total_gb,
                "gpu_percent": gpu_percent,
            }
            self._records.append(rec)

            # update tqdm postfix for live visibility
            if self._bar is not None:
                try:
                    headroom_gb = rec["sys_avail_gb"]
                    self._bar.set_postfix({
                        "job_rss_gb": f'{rec["job_rss_gb"]:.2f}',
                        "avail_gb": f'{headroom_gb:.2f}',
                        "swap%": f'{rec["swap_percent"]:.0f}',
                        "gpu_gb": f'{rec["gpu_used_gb"]:.2f}' if not pd.isna(rec["gpu_used_gb"]) else "NA",
                    })
                except Exception:
                    pass

            # warning when close to limits
            if rec["sys_avail_gb"] < 2.0 or rec["swap_percent"] > 50:
                # keep it silent—tqdm postfix already surfaces pressure;
                # if you prefer logs, hook your logger here
                pass

            self._stop.wait(self.interval_s)
