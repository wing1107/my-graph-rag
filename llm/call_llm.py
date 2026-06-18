#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
call_llm.py — LLM 统一调用入口（langchain 0.2.x / openai 1.x）

openai 0.28.x → 1.x 变更：
    openai.ChatCompletion.create(...)
    → openai.chat.completions.create(...)
"""

import json
import requests
import _thread as thread
import base64
import hashlib
import hmac
import os
import queue
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlparse, urlencode
from wsgiref.handlers import format_date_time

import zhipuai
import websocket
from dotenv import load_dotenv, find_dotenv


def get_completion(
    prompt: str,
    model: str,
    temperature=0.1,
    api_key=None,
    secret_key=None,
    access_token=None,
    appid=None,
    api_secret=None,
    max_tokens=2048,
):
    if model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k"]:
        return get_completion_gpt(prompt, model, temperature, api_key, max_tokens)
    elif model in ["ERNIE-Bot", "ERNIE-Bot-4", "ERNIE-Bot-turbo"]:
        return get_completion_wenxin(prompt, model, temperature, api_key, secret_key)
    elif model in ["Spark-1.5", "Spark-2.0"]:
        return get_completion_spark(prompt, model, temperature, api_key, appid, api_secret, max_tokens)
    elif model in [
        "chatglm_pro", "chatglm_std", "chatglm_lite",
        "glm-4", "glm-4-flash", "glm-4-air", "glm-4-airx",
        "glm-3-turbo", "glm-4-plus", "glm-4-long",
    ]:
        return get_completion_glm(prompt, model, temperature, api_key, max_tokens)
    elif model in ["qwen3-max", "qwen-flash", "qwen-plus", "qwen-long"]:
        return get_completion_qwen(prompt, model, temperature, api_key, max_tokens)
    else:
        return "不正确的模型"


def get_completion_gpt(prompt: str, model: str, temperature: float, api_key: str, max_tokens: int):
    """openai 1.x SDK 调用方式。"""
    import openai
    if api_key is None:
        api_key = parse_llm_api_key("openai")
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def get_access_token(api_key, secret_key):
    url = (
        f"https://aip.baidubce.com/oauth/2.0/token"
        f"?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}"
    )
    payload = json.dumps("")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")


def get_completion_wenxin(prompt: str, model: str, temperature: float, api_key: str, secret_key: str):
    if api_key is None or secret_key is None:
        api_key, secret_key = parse_llm_api_key("wenxin")
    access_token = get_access_token(api_key, secret_key)
    url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token={access_token}"
    payload = json.dumps({"messages": [{"role": "user", "content": "{}".format(prompt)}]})
    headers = {"Content-Type": "application/json"}
    response = requests.request("POST", url, headers=headers, data=payload)
    js = json.loads(response.text)
    return js["result"]


def get_completion_spark(
    prompt: str, model: str, temperature: float,
    api_key: str, appid: str, api_secret: str, max_tokens: int,
):
    if api_key is None or appid is None and api_secret is None:
        api_key, appid, api_secret = parse_llm_api_key("spark")
    if model == "Spark-1.5":
        domain = "general"
        Spark_url = "ws://spark-api.xf-yun.com/v1.1/chat"
    else:
        domain = "generalv2"
        Spark_url = "ws://spark-api.xf-yun.com/v2.1/chat"
    question = [{"role": "user", "content": prompt}]
    return spark_main(appid, api_key, api_secret, Spark_url, domain, question, temperature, max_tokens)


def get_completion_glm(prompt: str, model: str, temperature: float, api_key: str, max_tokens: int):
    if api_key is None:
        api_key = parse_llm_api_key("zhipuai")
    if hasattr(zhipuai, "ZhipuAI"):
        client = zhipuai.ZhipuAI(api_key=api_key, timeout=60)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    zhipuai.api_key = api_key
    response = zhipuai.model_api.invoke(
        model=model,
        prompt=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response["data"]["choices"][0]["content"].strip('"').strip(" ")


def get_completion_qwen(prompt: str, model: str, temperature: float, api_key: str, max_tokens: int):
    from http import HTTPStatus
    try:
        from dashscope import Generation
    except ImportError as exc:
        raise ImportError("dashscope not installed: pip install dashscope") from exc

    if api_key is None:
        api_key = parse_llm_api_key("qwen")

    response = Generation.call(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        result_format="message",
    )
    if getattr(response, "status_code", HTTPStatus.OK) != HTTPStatus.OK:
        raise RuntimeError(
            f"DashScope 调用失败: status={getattr(response, 'status_code', None)}, "
            f"message={getattr(response, 'message', None)}"
        )
    return response.output.choices[0].message.content.strip()


# ── 星火 WebSocket ────────────────────────────────────────────────────────
answer = ""


class Ws_Param(object):
    def __init__(self, APPID, APIKey, APISecret, Spark_url):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.host = urlparse(Spark_url).netloc
        self.path = urlparse(Spark_url).path
        self.Spark_url = Spark_url
        self.temperature = 0
        self.max_tokens = 2048

    def create_url(self):
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"
        signature_sha = hmac.new(
            self.APISecret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        signature_sha_base64 = base64.b64encode(signature_sha).decode("utf-8")
        authorization_origin = (
            f'api_key="{self.APIKey}", algorithm="hmac-sha256", '
            f'headers="host date request-line", signature="{signature_sha_base64}"'
        )
        authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode("utf-8")
        v = {"authorization": authorization, "date": date, "host": self.host}
        return self.Spark_url + "?" + urlencode(v)


def on_error(ws, error):
    print("### error:", error)


def on_close(ws, one, two):
    print(" ")


def on_open(ws):
    thread.start_new_thread(run, (ws,))


def run(ws, *args):
    data = json.dumps(
        gen_params(
            appid=ws.appid, domain=ws.domain, question=ws.question,
            temperature=ws.temperature, max_tokens=ws.max_tokens,
        )
    )
    ws.send(data)


def on_message(ws, message):
    data = json.loads(message)
    code = data["header"]["code"]
    if code != 0:
        print(f"请求错误: {code}, {data}")
        ws.close()
    else:
        choices = data["payload"]["choices"]
        status = choices["status"]
        content = choices["text"][0]["content"]
        print(content, end="")
        global answer
        answer += content
        if status == 2:
            ws.close()


def gen_params(appid, domain, question, temperature, max_tokens):
    return {
        "header": {"app_id": appid, "uid": "1234"},
        "parameter": {
            "chat": {
                "domain": domain,
                "random_threshold": 0.5,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "auditing": "default",
            }
        },
        "payload": {"message": {"text": question}},
    }


def spark_main(appid, api_key, api_secret, Spark_url, domain, question, temperature, max_tokens):
    output_queue = queue.Queue()

    def on_message(ws, message):
        data = json.loads(message)
        code = data["header"]["code"]
        if code != 0:
            ws.close()
        else:
            choices = data["payload"]["choices"]
            status = choices["status"]
            content = choices["text"][0]["content"]
            output_queue.put(content)
            if status == 2:
                ws.close()

    wsParam = Ws_Param(appid, api_key, api_secret, Spark_url)
    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = websocket.WebSocketApp(
        wsUrl, on_message=on_message, on_error=on_error,
        on_close=on_close, on_open=on_open,
    )
    ws.appid = appid
    ws.question = question
    ws.domain = domain
    ws.temperature = temperature
    ws.max_tokens = max_tokens
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    return "".join([output_queue.get() for _ in range(output_queue.qsize())])


def parse_llm_api_key(model: str, env_file: dict = None):
    """解析各厂商 API Key。"""
    if env_file is None:
        _ = load_dotenv(find_dotenv())
        env_file = os.environ

    def _get(key1, key2=None):
        v = env_file.get(key1) or (env_file.get(key2) if key2 else None)
        if not v:
            raise KeyError(f"环境变量 {key1} 未设置")
        return v

    if model == "openai":
        return _get("OPENAI_API_KEY")
    elif model == "wenxin":
        return _get("wenxin_api_key"), _get("wenxin_secret_key")
    elif model == "spark":
        return _get("spark_api_key"), _get("spark_appid"), _get("spark_api_secret")
    elif model == "zhipuai":
        return _get("ZHIPUAI_API_KEY", "zhipuai_api_key")
    elif model == "qwen":
        return _get("DASHSCOPE_API_KEY", "dashscope_api_key")
    else:
        raise ValueError(f"model {model} not supported")
