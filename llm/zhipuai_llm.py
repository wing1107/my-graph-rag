#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import logging
import os
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

try:
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
    )
except Exception:
    class CallbackManagerForLLMRun:
        def on_llm_new_token(self, token: str) -> None:
            return None

    class AsyncCallbackManagerForLLMRun:
        async def on_llm_new_token(self, token: str) -> None:
            return None


try:
    from langchain_core.outputs import GenerationChunk
except Exception:
    class GenerationChunk:
        def __init__(self, text: str):
            self.text = text


from llm.self_llm import Self_LLM

logger = logging.getLogger(__name__)


class ZhipuAILLM(Self_LLM):
    """兼容当前环境的智谱 AI 封装类。"""

    client: Any = None
    model: str = "glm-4-flash"
    zhipuai_api_key: Optional[str] = None
    incremental: Optional[bool] = True
    streaming: Optional[bool] = False
    request_timeout: Optional[int] = 60
    top_p: Optional[float] = 0.8
    temperature: Optional[float] = 0.95
    request_id: Optional[float] = None
    max_tokens: Optional[int] = 1024

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = kwargs.get("model", getattr(self, "model", "glm-4-flash"))
        self.zhipuai_api_key = kwargs.get("zhipuai_api_key") or os.getenv("ZHIPUAI_API_KEY")
        self.incremental = kwargs.get("incremental", getattr(self, "incremental", True))
        self.streaming = kwargs.get("streaming", getattr(self, "streaming", False))
        self.request_timeout = kwargs.get("request_timeout", getattr(self, "request_timeout", 60))
        self.top_p = kwargs.get("top_p", getattr(self, "top_p", 0.8))
        self.temperature = kwargs.get("temperature", getattr(self, "temperature", 0.95))
        self.request_id = kwargs.get("request_id", getattr(self, "request_id", None))
        self.max_tokens = kwargs.get("max_tokens", getattr(self, "max_tokens", 1024))
        self.model_kwargs = kwargs.get("model_kwargs", getattr(self, "model_kwargs", {})) or {}

        if not self.zhipuai_api_key:
            raise ValueError("请设置 ZHIPUAI_API_KEY 环境变量或传入 zhipuai_api_key")

        self.client = self._build_client()

    def _build_client(self) -> Any:
        try:
            import zhipuai
        except ImportError as exc:
            raise ValueError("zhipuai package not found, please install it with `pip install zhipuai`") from exc

        if hasattr(zhipuai, "ZhipuAI"):
            return zhipuai.ZhipuAI(api_key=self.zhipuai_api_key)

        zhipuai.api_key = self.zhipuai_api_key
        return zhipuai.model_api

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            **{"model": self.model},
            **super()._identifying_params,
        }

    @property
    def _llm_type(self) -> str:
        return "zhipuai"

    @property
    def _default_params(self) -> Dict[str, Any]:
        normal_params = {
            "streaming": self.streaming,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "request_id": self.request_id,
        }
        return {**normal_params, **self.model_kwargs}

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.streaming:
            completion = ""
            for chunk in self._stream(prompt, stop, run_manager, **kwargs):
                completion += chunk.text
            return completion

        if hasattr(self.client, "chat"):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content

        response = self.client.invoke(
            model=self.model,
            prompt=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return response["data"]["choices"][-1]["content"].strip('"').strip()

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return self._call(prompt, stop=stop, **kwargs)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
        chunk = GenerationChunk(text=text)
        yield chunk
        if run_manager:
            run_manager.on_llm_new_token(chunk.text)

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        chunk = GenerationChunk(text=self._call(prompt, stop=stop, **kwargs))
        yield chunk
        if run_manager:
            await run_manager.on_llm_new_token(chunk.text)
