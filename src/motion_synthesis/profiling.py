"""Retained local timing decorator; not used by the generation hot path."""

import time
from functools import wraps


def time_checker(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result

    return wrapper
