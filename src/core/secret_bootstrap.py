"""Startup shim: resolve the LLM API key via the SecretProvider seam and export
it to the environment for the SDK consumers that read it there.

Why a shim rather than a direct `secrets.get_secret()` call at each use site
(as INTEGRATION.md step 3 suggests):

  - Generation (generator.py) and judging (evaluator.py) both go through the
    Google Antigravity SDK, which reads the API key from the environment
    internally. There is no os.getenv() to replace and no per-call key
    parameter to inject.
  - Retrieval (retriever.py) reads GEMINI_API_KEY / GOOGLE_API_KEY from the
    environment directly.

So the single reliable injection point is: fetch the key once at startup and
set both env vars before any of those components run. Call bootstrap_secrets()
at each entry point after load_dotenv().

Backend selection lives in secret_provider.get_secret_provider():
  vault -> OS-keychain daemon; get_secret_provider() already fails loudly if
           the daemon is unreachable, so a missing key here is a hard error too.
  env   -> read from environment / .env (DEFAULT); tolerant if the key is
           absent, preserving the pre-existing "warn and continue" behavior
           (retriever.py already logs a warning and proceeds without a key).
"""

from __future__ import annotations

import logging
import os

from src.core.secret_provider import SecretNotFound, get_secret_provider

logger = logging.getLogger(__name__)

_bootstrapped = False


def bootstrap_secrets(provider_name: str = "gemini") -> bool:
    """Resolve the LLM key via the configured backend and export it to env.

    Sets both GEMINI_API_KEY and GOOGLE_API_KEY to the same value: the
    Antigravity SDK and the langchain Google embeddings each look for one of
    these, and we don't rely on knowing which. Idempotent across calls.

    Returns True if a key was exported. In SECRET_BACKEND=vault mode a missing
    key or unreachable daemon raises (loud, by design); in env mode it logs a
    warning and returns False so non-LLM work (and CI) can still proceed.
    """
    global _bootstrapped
    if _bootstrapped:
        return True

    backend = os.getenv("SECRET_BACKEND", "env").strip().lower()
    provider = get_secret_provider()  # enforces vault daemon health in vault mode

    # In env mode, accept either the gemini or google env var, matching what
    # retriever.py already tolerates. In vault mode, the vault is authoritative.
    names = [provider_name] if backend == "vault" else [provider_name, "google"]

    key = None
    last_exc: SecretNotFound | None = None
    for name in names:
        try:
            key = provider.get_secret(name)
            break
        except SecretNotFound as exc:
            last_exc = exc

    if key is None:
        if backend == "vault":
            raise last_exc  # requested the vault explicitly; do not degrade silently
        logger.warning(
            "No LLM key from SECRET_BACKEND=%s (%s). LLM calls will fail until "
            "a key is available.",
            backend,
            last_exc,
        )
        return False

    os.environ["GEMINI_API_KEY"] = key
    os.environ["GOOGLE_API_KEY"] = key
    _bootstrapped = True
    logger.info("Secrets bootstrapped via SECRET_BACKEND=%s.", backend)
    return True
