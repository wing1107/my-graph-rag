from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from pydantic.v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)

class ZhipuAIEmbeddings(BaseModel, Embeddings):
    """`ZhipuAI Embeddings` - 适配新版 SDK 2.0+"""

    zhipuai_api_key: Optional[str] = None
    client: Any = None
    model: str = "embedding-2"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        zhipuai_api_key = get_from_dict_or_env(
            values,
            "zhipuai_api_key",
            "ZHIPUAI_API_KEY",
        )
        values["zhipuai_api_key"] = zhipuai_api_key

        try:
            from zhipuai import ZhipuAI
            values["client"] = ZhipuAI(api_key=zhipuai_api_key)
        except ImportError:
            raise ValueError(
                "ZhipuAI package not found, please install it with "
                "`pip install zhipuai`"
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize ZhipuAI client: {e}")

        return values

    def _embed(self, text: str) -> List[float]:
        try:
            resp = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return resp.data[0].embedding
        except Exception as e:
            raise ValueError(f"Error raised by ZhipuAI API: {e}")

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("ZhipuAI Embeddings 暂不支持异步请求")

    async def aembed_query(self, text: str) -> List[float]:
        raise NotImplementedError("ZhipuAI Embeddings 暂不支持异步请求")