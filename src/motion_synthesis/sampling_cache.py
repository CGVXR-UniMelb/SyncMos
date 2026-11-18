"""Shared intermediate diffusion cache for both time-warp paths."""

from collections import OrderedDict


class diffusion_sampling_cache:
    """cache for FCS"""

    def __init__(self, max_cache_size=1000):
        self.cache = OrderedDict()
        self.max_cache_size = max_cache_size

    def get_sampling_information(self, current_step):
        """get information from cache"""
        assert lambda current_step: isinstance(current_step, int), "current_step should be an integer"
        cache_key = f"{current_step}"

        if cache_key in self.cache:
            # LRU update
            information = self.cache.pop(cache_key)
            self.cache[cache_key] = information
            return information
        else:
            raise ValueError(f"Sampling information for step {current_step} not found in cache")

    def store_sampling_information(self, current_step, info):
        """store sampling information to cache"""
        cache_key = f"{current_step}"

        # manage cache size
        if len(self.cache) >= self.max_cache_size:
            self.cache.popitem(last=False)  # remove the oldest item

        self.cache[cache_key] = info

    def invalidate_cache(self, from_step):
        """invalidate cache from certain step"""
        keys_to_remove = [k for k in self.cache.keys() if int(k) >= from_step]
        for key in keys_to_remove:
            del self.cache[key]

    def erase_all(self):
        """erase all cache"""
        self.cache.clear()

    def get_len(self):
        """get length of cache"""
        return len(self.cache)
