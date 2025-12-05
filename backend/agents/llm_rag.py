from typing import Any, Dict, List, Optional
import time
import os
import requests
from .ingest import Ingestor


class LLMAgent:
    """LLM + RAG agent using sentence-transformers + FAISS for retrieval and Ollama for generation.

    Features:
      - ingest_documents(docs): add docs to index
      - handle_request(..., action='query') -> runs retrieval + prompt + Ollama call
    """

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", index_path: Optional[str] = None):
        self.embedding_model_name = embedding_model_name
        self.index_path = index_path
        self._ingestor: Optional[Ingestor] = None

    def _ensure_ingestor(self):
        if self._ingestor is None:
            self._ingestor = Ingestor(model_name=self.embedding_model_name, index_path=self.index_path)

    def ingest_documents(self, docs: List[Dict[str, Any]], persist: bool = False):
        """Docs: list of {id, text, meta}
        """
        self._ensure_ingestor()
        self._ingestor.ingest(docs, persist=persist)

    def _build_prompt(self, question: str, retrieved: List[Dict[str, Any]]) -> str:
        # Simple prompt template: include top-K retrieved passages as context
        blocks = [f"Context {i+1}: {d['text']}\nSource: {d.get('meta', {})}" for i, (d, _) in enumerate(retrieved)]
        context_str = "\n\n".join(blocks)
        prompt = f"You are an assistant. Use the following context to answer the question.\n\n{context_str}\n\nQuestion: {question}\nAnswer concisely and include citations."
        return prompt

    def _call_ollama(self, prompt: str, model: str = "gpt-oss:20b-cloud") -> Dict[str, Any]:
        url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
        payload = {"model": model, "prompt": prompt}
        r = requests.post(url, json=payload, timeout=30.0)
        r.raise_for_status()
        return r.json()

    async def handle_request(self, session_id: str, action: str, args: Dict[str, Any], context: Dict[str, Any]):
        if action in ("query", "chat", None):
            question = args.get("question") or args.get("q") or args.get("prompt")
            if not question:
                raise ValueError("LLMAgent requires 'question' in args")

            # retrieve
            retrieved = []
            if self._ingestor is not None:
                retrieved = self._ingestor.query(question, top_k=5)

            prompt = self._build_prompt(question, retrieved)
            # call Ollama synchronously (blocking) but wrapped in async
            loop = None
            try:
                loop = __import__('asyncio').get_running_loop()
            except Exception:
                loop = None

            if loop and loop.is_running():
                # run in thread pool
                import asyncio
                res = await asyncio.get_running_loop().run_in_executor(None, self._call_ollama, prompt)
            else:
                res = self._call_ollama(prompt)

            return {"answer": res, "retrieved": [r[0] for r in retrieved]}

        raise ValueError(f"Unsupported LLMAgent action: {action}")
