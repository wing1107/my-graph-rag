#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""阿里云通义千问（DashScope）LangChain 适配类。

通过官方 dashscope SDK 调用千问系列模型（qwen3-max / qwen-flash / qwen3-plus / qwen3-long 等），
对外暴露与项目其它 *_llm.py 模块一致的 LangChain LLM 接口，便于在 RAG 链路中复用。
"""

import logging
import os
from http import HTTPStatus
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


class QwenLLM(Self_LLM):
    """阿里云千问 LLM 封装类（基于 dashscope SDK）。"""

    model: str = "qwen3-max"
    dashscope_api_key: Optional[str] = None
    streaming: Optional[bool] = False
    request_timeout: Optional[int] = 60
    top_p: Optional[float] = 0.8
    temperature: Optional[float] = 0.95
    max_tokens: Optional[int] = 1500

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = kwargs.get("model", getattr(self, "model", "qwen3-max"))
        self.dashscope_api_key = (
            kwargs.get("dashscope_api_key")
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("dashscope_api_key")
        )
        self.streaming = kwargs.get("streaming", getattr(self, "streaming", False))
        self.request_timeout = kwargs.get("request_timeout", getattr(self, "request_timeout", 60))
        self.top_p = kwargs.get("top_p", getattr(self, "top_p", 0.8))
        self.temperature = kwargs.get("temperature", getattr(self, "temperature", 0.95))
        self.max_tokens = kwargs.get("max_tokens", getattr(self, "max_tokens", 1500))
        self.model_kwargs = kwargs.get("model_kwargs", getattr(self, "model_kwargs", {})) or {}

        if not self.dashscope_api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量或传入 dashscope_api_key")

        # 校验 SDK 是否安装；调用入口在 _call 中通过 Generation.call(...) 实现
        try:
            import dashscope  # noqa: F401
        except ImportError as exc:
            raise ValueError(
                "dashscope package not found, please install it with `pip install dashscope`"
            ) from exc

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            **{"model": self.model},
            **super()._identifying_params,
        }

    @property
    def _llm_type(self) -> str:
        return "qwen"

    @property
    def _default_params(self) -> Dict[str, Any]:
        normal_params = {
            "streaming": self.streaming,
            "top_p": self.top_p,
            "temperature": self.temperature,
        }
        return {**normal_params, **self.model_kwargs}

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        from dashscope import Generation

        if self.streaming:
            completion = ""
            for chunk in self._stream(prompt, stop, run_manager, **kwargs):
                completion += chunk.text
            return completion

        response = Generation.call(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            api_key=self.dashscope_api_key,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            result_format="message",
        )

        if getattr(response, "status_code", HTTPStatus.OK) != HTTPStatus.OK:
            raise RuntimeError(
                f"DashScope 调用失败: request_id={getattr(response, 'request_id', None)}, "
                f"status_code={getattr(response, 'status_code', None)}, "
                f"code={getattr(response, 'code', None)}, "
                f"message={getattr(response, 'message', None)}"
            )

        # result_format="message" 时，输出结构与 OpenAI ChatCompletion 类似
        return response.output.choices[0].message.content

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
