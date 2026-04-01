"""One-time logging / env defaults: less HuggingFace & transformers chatter on the API process."""

from __future__ import annotations

import logging
import os


def _env_truthy(key: str) -> bool:
    return (os.getenv(key) or "").strip().lower() in ("1", "true", "yes")


def install_default_log_tuning() -> None:
    """
    Quiets Hugging Face Hub auth tips, transformers weight banners, and tqdm-style load spam
    unless ``HYEAERO_VERBOSE_MODEL_LOAD=1``.

    To **fix** the Hub warning properly (and raise rate limits), set ``HF_TOKEN`` to a read token
    from https://huggingface.co/settings/tokens — we only silence log noise here, not the need for a token in CI.
    """
    if _env_truthy("HYEAERO_VERBOSE_MODEL_LOAD"):
        return

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    # Auth nag + download hints (often WARNING on huggingface_hub)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)

    # "Loading weights...", XLMRoberta LOAD REPORT, etc.
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)

    # sentence-transformers can be chatty at INFO
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    # Ask Consultant timing lines — ensure visible if root is WARNING in some deployments
    logging.getLogger("rag.consultant_progress").setLevel(logging.INFO)
