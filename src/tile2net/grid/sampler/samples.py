import pandas as pd

from ..frame import FrameWrapper
from .. import frame

class Samples(
    FrameWrapper
):
    @frame.column
    def t(self):
        """time elapsed during sample"""

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
        """percent of swap used [0..100]"""

    @frame.column
    def gpu_used_gb(self) -> pd.Series:
        """GPU VRAM used (GB); may contain NaN if no CUDA"""

    @frame.column
    def gpu_total_gb(self) -> pd.Series:
        """total GPU VRAM (GB); may contain NaN if no CUDA"""

    @frame.column
    def gpu_percent(self) -> pd.Series:
        """percent of GPU VRAM used [0..100]; may contain NaN"""


