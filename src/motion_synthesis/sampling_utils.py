"""SyncMOS sampling helpers; kept unchanged when separating the bundled LINGO tree."""

import inspect

from dtw import dtw


def extract_ver2(a, t, x_shape):
    batch_size = t.shape[0]
    # Ensure t is on the same device as a for gather operation
    t_device = t.to(a.device)
    out = a.gather(-1, t_device)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def debug_assert(condition, message):
    if not condition:
        frame = inspect.currentframe().f_back
        filename = frame.f_code.co_filename.split("/")[-1]
        line_number = frame.f_lineno
        raise AssertionError(f"[{filename}:{line_number}] {message}")


class DTW:
    def __init__(self, dist_func):
        self.dist_func = dist_func

    def change_dist_func(self, dist_func):
        self.dist_func = dist_func

    def compute_distance(self, x, y):
        d = dtw(x, y, self.dist_func)
        return d

    def plot_dtw(self, x, y):
        pass
