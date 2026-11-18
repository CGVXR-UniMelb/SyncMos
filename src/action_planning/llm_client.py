"""
OpenAI Compatible Interface for various providers.

* Supported providers and their API key env var
    - openai: OPENAI_API_KEY
    - groq, groq-native: GROQ_API_KEY
    - lepton: LEPTON_API_TOKEN
    - deepinfra: DEEPINFRA_API_TOKEN
    - anthropic: ANTHROPIC_API_KEY
    - cerebras: CEREBRAS_API_KEY
      (waitlisted for a free key)
    - gemini: GEMINI_API_KEY
    - qwen-local: Qwen2.5-VL local model
    - qwen-dashscope: Qwen models via DashScope API (DASHSCOPE_API_KEY)
"""

import os
import time
from typing import List, Dict, Any

from dotenv import load_dotenv


class NotSupportedProviderError(Exception):
    pass


class ModelRequiredInBaseUrlError(Exception):
    pass


load_dotenv()

# Cache the local Qwen model instance by using the model name for reuse, avoiding repeated loading of weights
_QWEN_LOCAL_CLIENTS = {}
_GPT_OSS_CLIENTS = {}


class QwenClient:
    """Qwen3 Local Model Client"""

    def __init__(self, model_name: str = "Qwen/Qwen3-8B", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Loading Qwen3 Model and Processor"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Loading Qwen3 model: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map=self.device,
            )

            print(f"Qwen3 model loading completed, device: {self.model.device}")

        except ImportError as e:
            raise ImportError(f"Please install the transformers-related dependencies: pip install transformers {e}")

    def chat_completions_create(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        # Generating Text Using apply_chat_template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Switches between thinking and non-thinking modes. Default is True.
        )

        # Converting Text into Model Input
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # For thinking mode (enable_thinking=True),
        # use Temperature=0.6, TopP=0.95, TopK=20, and MinP=0.
        # DO NOT use greedy decoding, as it can lead to performance degradation and endless repetitions.

        # For non-thinking mode (enable_thinking=False),
        # we suggest using Temperature=0.7, TopP=0.8, TopK=20, and MinP=0.

        # For supported frameworks, you can adjust the presence_penalty parameter between 0 and 2 to reduce endless repetitions.
        # However, using a higher value may occasionally result in language mixing and a slight decrease in model performance.

        # Generating Parameters
        max_new_tokens = kwargs.get("max_tokens", 32768)
        temperature = kwargs.get("temperature", 0.6)
        if temperature == 0.0:
            do_sample = False
        else:
            do_sample = True

        # Generating Text
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

        # Only the newly generated parts are shown
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        # Skip tokens through the final </think> marker when present.
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        response_text = content
        return {
            "choices": [{"message": {"content": response_text, "role": "assistant"}}],
            "usage": {
                "prompt_tokens": len(model_inputs.input_ids[0]),
                "completion_tokens": len(output_ids),
                "total_tokens": len(model_inputs.input_ids[0]) + len(output_ids),
            },
        }


class GptOssClient:
    """Hugging Face Pipeline Client, for… openai/gpt-oss-20b Text Models"""

    def __init__(self, model_name: str = "openai/gpt-oss-20b", device: str = "auto"):
        self.model_name = model_name or "openai/gpt-oss-20b"
        self.device = device
        self.pipe = self._build_pipeline()

    def _build_pipeline(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Preserve the existing compatibility check and ImportError for older Transformers.
            from transformers import AutoConfig, Mxfp4Config  # noqa: F401
        except ImportError as e:
            raise ImportError(f"Please install the transformers dependencies: pip install transformers {e}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype="auto",
        )

    def chat_completions_create(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        # Generating Text Using apply_chat_template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Converting Text into Model Input
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Generating Parameters
        max_new_tokens = kwargs.get("max_tokens", 16384)
        temperature = kwargs.get("temperature", 1.0)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
        }

        # Temperature control sampling
        if temperature == 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature

        # Generating Text
        generated_ids = self.model.generate(**model_inputs, **gen_kwargs)

        # Only the newly generated parts are shown
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        response_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return {
            "choices": [{"message": {"content": response_text, "role": "assistant"}}],
            "usage": {
                "prompt_tokens": len(model_inputs.input_ids[0]),
                "completion_tokens": len(output_ids),
                "total_tokens": len(model_inputs.input_ids[0]) + len(output_ids),
            },
        }


class LLMClient:
    def __init__(self, provider="openai", model=None, api_key=None):
        """
        Some provider requires model for encoding base_url
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key or self._get_api_key()
        self.base_url = self._get_base_url()
        self.client = self._get_client()

    def _get_api_key(self):
        if self.provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif self.provider == "deepinfra":
            return os.getenv("DEEPINFRA_API_TOKEN")
        elif self.provider in ("groq-native", "groq"):
            return os.getenv("GROQ_API_KEY")
        elif self.provider == "lepton":
            return os.getenv("LEPTON_API_TOKEN")
        elif self.provider == "cerebras":
            return os.getenv("CEREBRAS_API_KEY")
        elif self.provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")
        elif self.provider == "google":
            return os.getenv("GEMINI_API_KEY")
        elif self.provider == "qwen-dashscope":
            return os.getenv("DASHSCOPE_API_KEY")
        elif self.provider == "qwen-local":
            return None  # The local model does not require an API key
        elif self.provider == "gptoss":
            return None  # The local model does not require an API key

        raise NotSupportedProviderError(f"{self.provider} is not a supported provider.")

    def _get_base_url(self):
        if self.provider == "deepinfra":
            return "https://api.deepinfra.com/v1/openai"
        elif self.provider == "groq":
            return "https://api.groq.com/openai/v1"
        elif self.provider == "lepton":
            if not self.model:
                raise ModelRequiredInBaseUrlError(f"model is required for encoding base_url for {self.provider}.")
            return f"https://{self.model}.lepton.run/api/v1"
        elif self.provider == "cerebras":
            return "https://api.cerebras.ai/v1"
        elif self.provider == "qwen-dashscope":
            return "https://dashscope.aliyuncs.com/compatible-mode/v1"
        elif self.provider in ("openai", "groq-native", "anthropic", "google", "qwen-local", "gptoss"):
            return None

        raise NotSupportedProviderError(f"{self.provider} is not a supported provider.")

    def _get_client(self):
        """
        Use Native client for *-native providers.
        """
        if self.provider == "groq-native":
            import groq  # type: ignore[reportMissingImports]

            return groq.Groq(api_key=self.api_key)
        elif self.provider == "anthropic":
            import anthropic  # type: ignore[reportMissingImports]

            return anthropic.Anthropic(api_key=self.api_key)
        elif self.provider == "google":
            from google import genai  # type: ignore[reportMissingImports]

            return genai.Client(api_key=self.api_key)
        elif self.provider == "qwen-local":
            # Reuse the same model instance to avoid repeated loading
            global _QWEN_LOCAL_CLIENTS
            key = self.model or "default"
            if key in _QWEN_LOCAL_CLIENTS:
                return _QWEN_LOCAL_CLIENTS[key]
            client = QwenClient(model_name=self.model)
            _QWEN_LOCAL_CLIENTS[key] = client
            return client
        elif self.provider == "gptoss":
            # Use gpt from Hugging Face pipeline-oss Client
            global _GPT_OSS_CLIENTS
            key = self.model or "default"
            if key in _GPT_OSS_CLIENTS:
                return _GPT_OSS_CLIENTS[key]
            client = GptOssClient(model_name=self.model)
            _GPT_OSS_CLIENTS[key] = client
            return client
        elif self.provider in ["openai", "deepinfra", "groq", "cerebras", "lepton", "qwen-dashscope"]:
            import openai  # type: ignore[reportMissingImports]

            return openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

        raise NotSupportedProviderError(f"{self.provider} is not a supported provider.")

    def __getattr__(self, name):
        return getattr(self.client, name)

    def get_response(self, with_usage=False, max_retries=3, retry_delay=60, **kwargs):
        """
        Obtain LLM response with a rate-limiting retry mechanism

        Args:
            with_usage: Whether to return token usage information
            max_retries: Maximum number of retries
            retry_delay: Retry delay time (seconds)
            **kwargs: LLM request parameters
        """
        if self.provider == "anthropic":
            return self._get_response_anthropic(with_usage, **kwargs)
        elif self.provider == "google":
            return self._get_response_google(with_usage, **kwargs)
        elif self.provider == "qwen-local":
            return self._get_response_qwen(with_usage, **kwargs)
        elif self.provider == "gptoss":
            return self._get_response_gptoss(with_usage, **kwargs)

        # Handle rate-limiting retries
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                response = response.choices[0].message.content
                if with_usage:
                    return response, prompt_tokens, completion_tokens
                return response
            except Exception as e:
                # Check if it’s a rate-limiting error
                if "rate_limit" in str(e).lower() or "429" in str(e) or "tpm" in str(e).lower():
                    if attempt < max_retries:
                        print(
                            f"⚠️ Encountered rate limit, waiting for {retry_delay} seconds before retrying... (try {attempt + 1}/{max_retries + 1})"
                        )
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"❌ Maximum retries reached, rate limit error: {e}")
                        raise
                else:
                    # No rate-limiting error, simply thrown out
                    raise

    def _get_response_qwen(self, with_usage=False, **kwargs):
        """Handling response for Qwen’s local model"""
        response = self.client.chat_completions_create(**kwargs)
        prompt_tokens = response["usage"]["prompt_tokens"]
        completion_tokens = response["usage"]["completion_tokens"]
        response_text = response["choices"][0]["message"]["content"]

        if with_usage:
            return response_text, prompt_tokens, completion_tokens
        return response_text

    def _get_response_gptoss(self, with_usage=False, **kwargs):
        response = self.client.chat_completions_create(**kwargs)
        response_text = response["choices"][0]["message"]["content"]
        if with_usage:
            u = response.get("usage", {})
            return response_text, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        return response_text

    def _get_response_anthropic(self, with_usage=False, **kwargs):
        messages = kwargs.get("messages")
        if messages[0]["role"] == "system":
            system_message = messages[0]["content"]
            messages = messages[1:]
        else:
            system_message = None

        response = self.client.messages.create(
            model=kwargs.get("model"),
            system=system_message,
            messages=messages,
            temperature=kwargs.get("temperature"),
            max_tokens=2048,
        )
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens
        response = response.content[0].text
        if with_usage:
            return response, prompt_tokens, completion_tokens
        return response

    def _get_response_google(self, with_usage=False, **kwargs):
        from google.genai.types import Content, GenerateContentConfig, Part  # type: ignore[reportMissingImports]

        messages = kwargs.get("messages")
        if messages[0]["role"] == "system":
            system_message = messages[0]["content"]
            messages = messages[1:]
        else:
            system_message = None

        conversation_history = []
        for i, message in enumerate(messages[:-1]):  # the last message is the query
            if i % 2 == 0:
                assert message["role"] == "user"
            else:
                assert message["role"] == "assistant"

            conversation_history.append(
                Content(
                    parts=[Part(text=message["content"])],
                    role="user" if message["role"] == "user" else "model",
                )
            )

        response_format = kwargs.get("response_format", None)
        if response_format:
            assert response_format["type"] == "json_object"
            response_mime_type = "application/json"
        else:
            response_mime_type = None

        chat = self.client.chats.create(
            model=kwargs.get("model"),
            history=conversation_history,
            config=GenerateContentConfig(
                system_instruction=system_message,
                temperature=kwargs.get("temperature"),
                max_output_tokens=2048,
                response_mime_type=response_mime_type,
            ),
        )
        response = chat.send_message(messages[-1]["content"])

        prompt_tokens = response.usage_metadata.prompt_token_count
        completion_tokens = response.usage_metadata.candidates_token_count
        response = response.text
        if with_usage:
            return response, prompt_tokens, completion_tokens
        return response


def main():
    provider = "openai"
    model = "gpt-4o-mini"
    client = LLMClient(provider=provider, model=model)
    args = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a simple calculator. Respond with only the result as the JSON format. Do not include any additional text or explanations.",
            },
            {"role": "user", "content": "1+2"},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    (
        response,
        prompt_tokens,
        completion_tokens,
    ) = client.get_response(with_usage=True, **args)
    print(prompt_tokens)
    print(completion_tokens)
    print(response)


if __name__ == "__main__":
    main()
