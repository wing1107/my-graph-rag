#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from typing import Any, Dict, Mapping, Optional

try:
    from langchain_core.language_models.llms import LLM
except Exception:
    class LLM:
        """LangChain 不可用时的最小兼容基类。"""

        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

        def __call__(self, prompt: str, **kwargs: Any) -> str:
            return self._call(prompt, **kwargs)

        def invoke(self, prompt: str, **kwargs: Any) -> str:
            return self._call(prompt, **kwargs)


try:
    from pydantic import Field
except Exception:
    def Field(default=None, default_factory=None):
        if default_factory is not None:
            return default_factory()
        return default


class Self_LLM(LLM):
    # 自定义 LLM
    # 继承自 langchain.llms.base.LLM
    # 原生接口地址
    url: Optional[str] = None
    # 默认选用 GPT-3.5 模型，即目前一般所说的百度文心大模型
    model_name: str = "gpt-3.5-turbo"
    # 访问时延上限
    request_timeout: Optional[float] = None
    # 温度系数
    temperature: float = 0.1
    # API_Key
    api_key: Optional[str] = None
    # 必备的可选参数
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        cls = type(self)
        values = {
            # Read class defaults instead of instance attrs before BaseModel init.
            "url": getattr(cls, "url", None),
            "model_name": getattr(cls, "model_name", "gpt-3.5-turbo"),
            "request_timeout": getattr(cls, "request_timeout", None),
            "temperature": getattr(cls, "temperature", 0.1),
            "api_key": getattr(cls, "api_key", None),
            "model_kwargs": {},
        }
        values.update(kwargs)
        values["model_kwargs"] = values.get("model_kwargs") or {}
        super().__init__(**values)

    # 定义一个返回默认参数的方法
    @property
    def _default_params(self) -> Dict[str, Any]:
        """获取调用默认参数。"""
        normal_params = {
            "temperature": self.temperature,
            "request_timeout": self.request_timeout,
        }
        return {**normal_params}

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}
