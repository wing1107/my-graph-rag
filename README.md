# myGraphRagProject

基于 **LangGraph Self-RAG** 的本地知识库问答系统。

## 特性

- **Self-RAG 流程**：检索 → 文档评分 → 问题改写 → 生成 → 答案验证，支持自动循环重试
- **混合检索**：FAISS 向量检索 + BM25 关键词检索
- **多 Embedding**：m3e-base / multilingual-e5-large / ZhipuAI
- **多 LLM**：智谱 GLM-4 / 阿里云千问
- **会话持久化**：SQLite 多轮对话历史
- **双入口**：Gradio Web UI + FastAPI REST

## 快速开始

```powershell
# 1. 安装依赖
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. 配置 API Key
copy .env.example .env   # 填入 ZHIPUAI_API_KEY 或 DASHSCOPE_API_KEY

# 3. 构建向量库（首次使用）
python database/create_db.py

# 4. 启动
python run_gradio.py          # Gradio UI → http://127.0.0.1:7860
# 或
uvicorn api.fast_api:app --reload --port 8000   # REST API → http://localhost:8000/docs
```

## Self-RAG 流程

```
用户问题
  └─► retrieve → grade_documents ─[无相关文档]─► rewrite_query ─┐
                      │                                          └─► retrieve
                      └─► generate → grade_answer ─[幻觉]─► generate
                                          │
                                         END
```

## API

`POST /answer/`

```json
{
  "prompt": "什么是机器学习？",
  "model": "glm-4-flash",
  "embedding": "m3e",
  "session_id": "optional-session-id"
}
```

## 测试

```powershell
python -m pytest test/test_graph_nodes.py -v
```

## 技术栈

Python 3.10+ · LangChain 0.2 · LangGraph 0.2 · FAISS · BM25 · Gradio 4 · FastAPI

