"""Agent configuration and default registry.

Keep configuration simple and environment-friendly. Swap providers via environment
variables or by modifying this module during deployment.
"""

AGENT_REGISTRY = {
    "compute": "ComputeAgent",
    "llm.rag": "LLMAgent",
}

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_ON_DISK = False
