"""Answer formatting helpers for RAG / consultant."""

from rag.answer.aviation_formatter import (
    AVIATION_ANSWER_FORMAT_CONTRACT,
    AVIATION_ANSWER_SECTION_ORDER,
    AVIATION_SECTION_COMPARISON,
    AVIATION_SECTION_CONCLUSION,
    AVIATION_SECTION_OPERATIONAL,
    AVIATION_SECTION_SHORT_ANSWER,
    aviation_answer_format_contract_block,
)
from rag.answer.format import consultant_answer_style_suffix

__all__ = [
    "AVIATION_ANSWER_FORMAT_CONTRACT",
    "AVIATION_ANSWER_SECTION_ORDER",
    "AVIATION_SECTION_COMPARISON",
    "AVIATION_SECTION_CONCLUSION",
    "AVIATION_SECTION_OPERATIONAL",
    "AVIATION_SECTION_SHORT_ANSWER",
    "aviation_answer_format_contract_block",
    "consultant_answer_style_suffix",
]
