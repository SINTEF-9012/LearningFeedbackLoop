"""
LLM module - Language model integration.

This module contains:
- explainer: LLMExplainer for generating human-readable explanations
- rag: LLMAgent for retrieval-augmented generation
- ingest: Ingestor for document embedding and indexing
"""

from .explainer import LLMExplainer, ExplainerConfig
from .rag import LLMAgent
from .ingest import Ingestor

__all__ = [
    "LLMExplainer",
    "ExplainerConfig",
    "LLMAgent",
    "Ingestor",
]
