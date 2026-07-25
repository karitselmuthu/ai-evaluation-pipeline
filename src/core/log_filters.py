"""
Noise suppression for third-party log spam.

The Google Antigravity SDK (used by the RAG reranker/embeddings in
`retriever.py`) pipes its subprocess harness stderr into Python's root logger
at INFO level. When the Gemini project hits its monthly spend cap it emits a
*storm* of identical 429 / RESOURCE_EXHAUSTED / "retryable api error" lines
that bury the app's own output.

`install_noise_filter()` attaches a logging filter to the root logger that
drops those specific lines while leaving all real application logs intact.

Set the env var `VERBOSE_HARNESS_LOGS=1` to see the raw harness output again
(useful when debugging the SDK itself).
"""

import os
import logging

# Substrings that identify third-party harness / billing noise. Matching is
# case-insensitive. Kept deliberately narrow so real app logs are never dropped.
_NOISE_MARKERS = (
    "harness stderr",
    "spending cap",
    "spend cap",
    "resource_exhausted",
    "retryable api error",
    "retryable error",
    "model api is currently overloaded",
    "ai.studio",
    "gemini-api/docs/billing",
)

_installed = False


class _HarnessNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage().lower()
        except Exception:
            return True  # never let the filter itself drop a record on error
        return not any(marker in msg for marker in _NOISE_MARKERS)


def install_noise_filter() -> None:
    """
    Install the harness-noise filter on the root logger (idempotent).

    No-op if VERBOSE_HARNESS_LOGS is truthy, so the noise can be restored for
    debugging without code changes.
    """
    global _installed
    if _installed:
        return
    if os.getenv("VERBOSE_HARNESS_LOGS", "").strip().lower() in ("1", "true", "yes"):
        return

    noise_filter = _HarnessNoiseFilter()
    root = logging.getLogger()
    root.addFilter(noise_filter)
    # basicConfig may have already attached handlers; filter them too, because a
    # handler-level record can bypass a logger-level filter depending on setup.
    for handler in root.handlers:
        handler.addFilter(noise_filter)
    _installed = True
