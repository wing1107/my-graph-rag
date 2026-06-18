# langchain 0.2.x+：ChatOpenAI 已迁移到 langchain_openai
from langchain_openai import ChatOpenAI

from llm.call_llm import parse_llm_api_key
from llm.qwen_llm import QwenLLM
from llm.spark_llm import Spark_LLM
from llm.wenxin_llm import Wenxin_LLM
from llm.zhipuai_llm import ZhipuAILLM


def model_to_llm(
    model: str = None,
    temperature: float = 0.0,
    appid: str = None,
    api_key: str = None,
    Spark_api_secret: str = None,
    Wenxin_secret_key: str = None,
):
    """
    按模型名称构造 LangChain LLM 对象。

    星火：model, temperature, appid, api_key, api_secret
    百度文心：model, temperature, api_key, api_secret
    智谱：model, temperature, api_key
    OpenAI：model, temperature, api_key
    千问：model, temperature, api_key
    """
    if model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k"]:
        if api_key is None:
            api_key = parse_llm_api_key("openai")
        llm = ChatOpenAI(model_name=model, temperature=temperature, openai_api_key=api_key)

    elif model in ["ERNIE-Bot", "ERNIE-Bot-4", "ERNIE-Bot-turbo"]:
        if api_key is None or Wenxin_secret_key is None:
            api_key, Wenxin_secret_key = parse_llm_api_key("wenxin")
        llm = Wenxin_LLM(model=model, temperature=temperature, api_key=api_key, secret_key=Wenxin_secret_key)

    elif model in ["Spark-1.5", "Spark-2.0"]:
        if api_key is None or appid is None and Spark_api_secret is None:
            api_key, appid, Spark_api_secret = parse_llm_api_key("spark")
        llm = Spark_LLM(model=model, temperature=temperature, appid=appid, api_secret=Spark_api_secret, api_key=api_key)

    elif model in [
        "chatglm_pro", "chatglm_std", "chatglm_lite",
        "glm-4", "glm-4-flash", "glm-4-air", "glm-4-airx",
        "glm-3-turbo", "glm-4-plus", "glm-4-long",
    ]:
        if api_key is None:
            api_key = parse_llm_api_key("zhipuai")
        llm = ZhipuAILLM(model=model, zhipuai_api_key=api_key, temperature=temperature)

    elif model in ["qwen3-max", "qwen-flash", "qwen-plus", "qwen-long"]:
        if api_key is None:
            api_key = parse_llm_api_key("qwen")
        llm = QwenLLM(model=model, dashscope_api_key=api_key, temperature=temperature)

    else:
        raise ValueError(f"model {model} not supported")

    return llm
