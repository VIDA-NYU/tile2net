from functools import cached_property

import pandas as pd

from ..frame import FrameWrapper
from .. import frame

class Samples(
    FrameWrapper
):
    @frame.column
    def t(self):
        """timestamp (s) since epoch"""

    @frame.column
    def sys_used_gb(self) -> pd.Series:
        """total system memory used (GB)"""

    @frame.column
    def sys_avail_gb(self) -> pd.Series:
        """immediately available system memory (GB)"""

    @frame.column
    def sys_percent(self) -> pd.Series:
        """percent of system memory used [0..100]"""

    @frame.column
    def proc_rss_gb(self) -> pd.Series:
        """RSS of the current (parent) process (GB)"""

    @frame.column
    def children_rss_gb(self) -> pd.Series:
        """sum of RSS across dataloader worker processes (GB)"""

    @frame.column
    def job_rss_gb(self) -> pd.Series:
        """proc_rss_gb + children_rss_gb (GB)"""

    @frame.column
    def swap_used_gb(self) -> pd.Series:
        """swap space used (GB)"""

    @frame.column
    def swap_percent(self) -> pd.Series:
        """percent of swap space used [0..100]"""

    @frame.column
    def cpu_percent(self) -> pd.Series:
        """overall CPU utilization percent [0..100]"""

    @frame.column
    def cpu_cores(self) -> pd.Series:
        """logical CPU cores count"""

    @frame.column
    def gpu_used_gb(self) -> pd.Series:
        """GPU VRAM used (GB); may contain NaN if no CUDA"""

    @frame.column
    def gpu_total_gb(self) -> pd.Series:
        """total GPU VRAM (GB); may contain NaN if no CUDA"""

    @frame.column
    def gpu_percent(self) -> pd.Series:
        """percent of GPU VRAM used [0..100]; may contain NaN"""

    @frame.column
    def gpu_util_percent(self) -> pd.Series:
        """GPU compute utilization percent [0..100]; NaN if unavailable"""

    @frame.column
    def gpu_cores(self) -> pd.Series:
        """GPU multiprocessor (SM) count; NaN if unavailable"""

    # helper for safe aggregate computation on a named column
    def _safe_agg_col(self, key: str, how: str) -> float | None:
        df = getattr(self, 'frame', None)
        if df is None or key not in df:
            return None
        s = df[key]
        if getattr(s, 'empty', False):
            return None
        if hasattr(s, 'dropna') and s.dropna().empty:
            return None
        if how == 'max':
            v = s.max()
        elif how == 'mean':
            v = s.mean()
        else:
            raise ValueError(how)
        try:
            if v != v:
                return None
            return float(v)
        except Exception:
            return None

    @cached_property
    def max_vram(self) -> float | None:
        """maximum GPU VRAM percent used [0..100]; NaN if no CUDA"""
        return self._safe_agg_col('gpu_percent', 'max')

    @cached_property
    def avg_vram(self) -> float | None:
        """average GPU VRAM percent used [0..100]; NaN if no CUDA"""
        return self._safe_agg_col('gpu_percent', 'mean')

    @cached_property
    def max_gpu(self) -> float | None:
        """maximum GPU compute utilization percent [0..100] during sampling"""
        return self._safe_agg_col('gpu_util_percent', 'max')

    @cached_property
    def avg_gpu(self) -> float | None:
        """average GPU compute utilization percent [0..100] during sampling"""
        return self._safe_agg_col('gpu_util_percent', 'mean')

    @cached_property
    def max_ram(self) -> float | None:
        """maximum system RAM percent used [0..100] during sampling"""
        return self._safe_agg_col('sys_percent', 'max')

    @cached_property
    def avg_ram(self) -> float | None:
        """average system RAM percent used [0..100] during sampling"""
        return self._safe_agg_col('sys_percent', 'mean')

    @cached_property
    def max_cpu(self) -> float | None:
        """maximum CPU utilization percent [0..100] during sampling"""
        return self._safe_agg_col('cpu_percent', 'max')

    @cached_property
    def avg_cpu(self) -> float | None:
        """average CPU utilization percent [0..100] during sampling"""
        return self._safe_agg_col('cpu_percent', 'mean')

    @cached_property
    def time_elapsed(self) -> float | None:
        """overall time elapsed (s) during sampling window"""
        df = getattr(self, 'frame', None)
        if df is None or 't' not in df or df['t'].empty:
            return None
        try:
            v = float(df['t'].max() - df['t'].min())
        except Exception:
            return None
        if v != v:
            return None
        return v


