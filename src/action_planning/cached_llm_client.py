"""
LLMClient with caching functionality, inherits from LLMClient
It can save input and output data, allowing for direct retrieval of cached output when the same input is provided next time
"""

import os
import json
import hashlib
from typing import List, Dict, Any, Optional
import copy
from .llm_client import LLMClient


class CachedLLMClient(LLMClient):
    """LLM client with caching functionality"""

    def __init__(self, provider="openai", model=None, api_key=None, cache_dir="cache/llm_cache", enable_cache=True):
        """
        Initialize the caching LLM client

        Args:
            provider: LLM provider
            model: Model name
            api_key: API key
            cache_dir: Caching directory
            enable_cache: Whether caching functionality is enabled
        """
        super().__init__(provider, model, api_key)
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir
        if self.enable_cache:
            self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        """Ensure the caching directory exists"""
        os.makedirs(self.cache_dir, exist_ok=True)

    def _normalize_messages_for_cache(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Copy the message and replace the data URL of the embedded image with a MD5 digest placeholder.

        The digest uses the complete data URL; changes in image content or encoding will still affect the caching key.
        The original message from the caller remains unchanged for regular remote URLs.
        """
        if not messages:
            return messages
        msgs = copy.deepcopy(messages)
        for m in msgs:
            content = m.get("content")
            # Vision mode: content may be a list[ {type:'text'},{type:'image_url', image_url:{url:..., detail:...}} ]
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        img = item.get("image_url", {})
                        url = img.get("url")
                        if isinstance(url, str) and url.startswith("data:image"):
                            # Generate a stable hash placeholder using a data URL
                            digest = hashlib.md5(url.encode("utf-8")).hexdigest()
                            item["image_url"]["url"] = f"<IMAGE_BASE64_MD5:{digest}>"
            # Plain text mode: content is a string, no processing required
        return msgs

    def _get_cache_key(self, **kwargs) -> str:
        """
        Generate a cache key based on input parameters

        Args:
            **kwargs: LLM request parameters

        Returns:
            MD5 hash of the cache key
        """
        # Create a dictionary containing all parameters for generating the cache key
        # Shorten the cache key input with a summary of the embedded image data URL, preserving the image content differences.
        normalized_messages = self._normalize_messages_for_cache(kwargs.get("messages", []))

        cache_data = {
            "provider": self.provider,
            "model": self.model,
            "messages": normalized_messages,
            "temperature": kwargs.get("temperature", 1.0),
            "max_tokens": kwargs.get("max_tokens"),
            "response_format": kwargs.get("response_format"),
            # More parameters can be added
        }

        # Remove None values
        cache_data = {k: v for k, v in cache_data.items() if v is not None}

        # Convert the dictionary to a JSON string and generate a MD5 hash
        cache_str = json.dumps(cache_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(cache_str.encode("utf-8")).hexdigest()

    def _get_cache_file_path(self, cache_key: str) -> str:
        """Obtain the cache file path"""
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Load the response from the cache

        Args:
            cache_key: Cache key

        Returns:
            cached response data; returns None if not found
        """
        cache_file = self._get_cache_file_path(cache_key)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Unable to read cache file {cache_file}: {e}")
                return None
        return None

    def _save_to_cache(self, cache_key: str, response_data: Dict[str, Any]):
        """
        Save response to cache

        Args:
            cache_key: Cache key
            response_data: Response data
        """
        cache_file = self._get_cache_file_path(cache_key)
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(response_data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"Warning: Unable to save cache file {cache_file}: {e}")

    def get_response(self, with_usage=False, max_retries=3, retry_delay=60, **kwargs):
        """
        Obtain LLM response, prioritize using cache with rate-limiting retry mechanism

        Args:
            with_usage: Whether to return token usage information
            max_retries: Maximum number of retries
            retry_delay: Retry delay time (seconds)
            **kwargs: LLM request parameters

        Returns:
            LLM response content, if with_usage=Return True if true(response, prompt_tokens, completion_tokens)
        """
        # If caching is disabled, directly call the parent class method
        if not self.enable_cache:
            return super().get_response(
                with_usage=with_usage, max_retries=max_retries, retry_delay=retry_delay, **kwargs
            )

        # Generate a cache key
        cache_key = self._get_cache_key(**kwargs)

        # Attempt to load from the cache
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            print(f"Using cached response (Cache key: {cache_key[:8]}...)")
            if with_usage:
                return (
                    cached_data["response"],
                    cached_data.get("prompt_tokens", 0),
                    cached_data.get("completion_tokens", 0),
                )
            return cached_data["response"]

        # No cache hit, call the parent class method to get a response
        print(f"Cache miss, calling LLM API (Cache key: {cache_key[:8]}...)")
        if with_usage:
            response, prompt_tokens, completion_tokens = super().get_response(
                with_usage=True, max_retries=max_retries, retry_delay=retry_delay, **kwargs
            )
            # Save to the cache
            cache_data = {
                "response": response,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cache_key": cache_key,
                "provider": self.provider,
                "model": self.model,
            }
            self._save_to_cache(cache_key, cache_data)
            return response, prompt_tokens, completion_tokens
        else:
            response = super().get_response(
                with_usage=False, max_retries=max_retries, retry_delay=retry_delay, **kwargs
            )
            # Save to the cache
            cache_data = {"response": response, "cache_key": cache_key, "provider": self.provider, "model": self.model}
            self._save_to_cache(cache_key, cache_data)
            return response

    def clear_cache(self):
        """Clear all caches"""
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".json"):
                    os.remove(os.path.join(self.cache_dir, filename))
            print(f"Cache directory cleared: {self.cache_dir}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dictionary containing the number of cache files and total size
        """
        if not os.path.exists(self.cache_dir):
            return {"file_count": 0, "total_size": 0}

        files = [f for f in os.listdir(self.cache_dir) if f.endswith(".json")]
        total_size = sum(os.path.getsize(os.path.join(self.cache_dir, f)) for f in files)

        return {"file_count": len(files), "total_size": total_size, "cache_dir": self.cache_dir}


def main():
    """TestCachedLLMClient"""
    # Create a cache client
    client = CachedLLMClient(provider="openai", model="gpt-4o-mini")

    # Test Parameters
    args = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are a simple calculator. Only return the calculation result; no additional text or explanations are required.",
            },
            {"role": "user", "content": "1+What is 2 equal to?"},
        ],
        "temperature": 0.0,
    }

    print("First call (API should be called):")
    response1 = client.get_response(**args)
    print(f"Response: {response1}")

    print("\n Second call (cache should be used):")
    response2 = client.get_response(**args)
    print(f"Response: {response2}")

    print(f"\n cache statistics: {client.get_cache_stats()}")


if __name__ == "__main__":
    main()
