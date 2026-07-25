"""LLM clients for the evaluation pipeline (vault-proxy path — ACTIVE).

Two distinct call paths, deliberately different:

  GeneratorClient - may use the vault proxy's fallback rotation. If a provider
                    rate-limits, another can serve the call. Resilience matters
                    more than provider uniformity here.

  JudgeClient     - PINNED to one provider, fallback DISABLED. Scores must be
                    comparable across every case in a run; if the judge silently
                    changes model mid-run, the scores stop meaning one thing and
                    the comparison is corrupted.

Both record which provider actually served each call, so a report can never
hide a mixed-provider run. Talks to the local vault daemon's OpenAI-compatible
proxy at VAULT_URL (default http://127.0.0.1:9876); the daemon resolves the
real provider key from the OS keychain, so keys never enter this process.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import requests

from src.core.secret_provider import SecretProvider, get_secret_provider

# Body keys some upstreams stamp on the response, e.g. groq -> "x_groq".
# Used for provider attribution because the proxy emits no X-Serving-Provider
# header today (see INTEGRATION.md "Serving-provider detection").
_PROVIDER_BODY_TELLS = ("groq", "mistral", "openrouter", "openai", "anthropic", "gemini", "google")


@dataclass
class CallResult:
    """One model call, with attribution of who actually served it."""

    text: str
    requested_provider: str          # what we asked for
    serving_provider: str            # what actually answered (may differ if fallback)
    model: str
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    fell_back: bool = False          # True if serving != requested
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_record(self) -> dict[str, Any]:
        """Flat dict for the aggregator / report. Excludes the raw payload."""
        d = asdict(self)
        d.pop("raw", None)
        return d


class _BaseClient:
    """Shared plumbing for talking to the vault's OpenAI-compatible proxy."""

    def __init__(
        self,
        provider: str,
        model: str = "default",
        secret_provider: SecretProvider | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.provider = provider
        self.model = model
        self.secrets = secret_provider or get_secret_provider()
        self.base_url = (base_url or os.getenv("VAULT_URL", "http://127.0.0.1:9876")).rstrip("/")
        self.timeout = timeout

    def _auth_header(self) -> dict[str, str]:
        """Vault proxy authorizes callers with the session token.

        The provider API key is resolved inside the daemon from the OS keychain,
        so it never enters this process. We only need the session token.
        """
        token_path = Path(os.path.expanduser("~/.llm_vault_token"))
        try:
            token = token_path.read_text().strip()
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Vault session token not found at {token_path}. "
                f"Is the vault daemon running?"
            ) from exc
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    @staticmethod
    def _extract_serving_provider(resp: requests.Response, payload: dict) -> str:
        """Determine which provider actually served the call.

        Checked in order:
          1. an explicit response header (preferred - add this to the daemon)
          2. a field in the JSON body
          3. a provider-specific body key like "x_groq"
          4. the model string, which often carries a provider prefix
        Returns 'unknown' rather than guessing wrong - an honest 'unknown' in
        the report is better than a confident mislabel.
        """
        hdr = resp.headers.get("X-Serving-Provider") or resp.headers.get("x-serving-provider")
        if hdr:
            return hdr.strip()
        for key in ("served_by", "provider", "serving_provider"):
            if key in payload:
                return str(payload[key])
        for prov in _PROVIDER_BODY_TELLS:
            if f"x_{prov}" in payload:
                return prov
        model = str(payload.get("model", ""))
        if "/" in model:                      # e.g. "openrouter/anthropic/claude-..."
            return model.split("/", 1)[0]
        return "unknown"

    def _post(self, messages: list[dict], **kwargs: Any) -> CallResult:
        body = {"model": self.model, "messages": messages, **kwargs}
        started = time.perf_counter()
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self._auth_header(),
            json=body,
            timeout=self.timeout,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        resp.raise_for_status()
        data = resp.json()

        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        serving = self._extract_serving_provider(resp, data)

        return CallResult(
            text=text,
            requested_provider=self.provider,
            serving_provider=serving,
            model=str(data.get("model", self.model)),
            latency_ms=latency_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            fell_back=(serving not in ("unknown", self.provider)),
            raw=data,
        )


class GeneratorClient(_BaseClient):
    """Generates the answers being evaluated. Fallback rotation is ALLOWED.

    A long eval run makes many calls and 429s are the normal failure mode;
    letting the proxy rotate keeps the run alive. The cost is that a run may
    mix providers - which is why every CallResult records serving_provider and
    fell_back, so the report shows it instead of hiding it.

    NOTE: if you are comparing providers against each other, do NOT rely on
    fallback - pin each arm of the comparison explicitly.
    """

    def generate(self, messages: list[dict], **kwargs: Any) -> CallResult:
        return self._post(messages, **kwargs)


class JudgeClient(_BaseClient):
    """Scores outputs. PINNED to one provider; fallback is DISABLED.

    Scoring consistency is the whole basis of the comparison. If the judge
    silently switches models mid-run, early and late cases are graded by
    different graders and the aggregate is meaningless. So a fallback here is
    treated as a hard error, not a recovery.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.model in ("default", "auto"):
            raise ValueError(
                "JudgeClient requires an explicit model - 'default'/'auto' let "
                "the proxy choose, which breaks scoring consistency."
            )

    def judge(self, messages: list[dict], **kwargs: Any) -> CallResult:
        result = self._post(messages, **kwargs)

        # The proxy has no "no-rotation" flag today, so we enforce it after the
        # fact: if a judge call was served by a different provider, the run's
        # scores are no longer comparable. Treat as a hard error, not recovery.
        if result.fell_back:
            raise RuntimeError(
                f"Judge call fell back from '{result.requested_provider}' to "
                f"'{result.serving_provider}'. Scores across this run would not "
                f"be comparable. Fix the rate limit or pin a reachable judge "
                f"provider, then re-run - do not accept mixed-judge results."
            )
        return result


def provider_summary(results: list[CallResult]) -> dict[str, Any]:
    """Roll up provider attribution for the run report.

    Surfaces mixed-provider runs explicitly so a reader of the report can see
    that a comparison may be affected, rather than reading a single average
    that quietly spans two models.
    """
    return summarize_providers(
        [(r.serving_provider, r.fell_back) for r in results]
    )


def summarize_providers(serving_and_fellback: list[tuple[str, bool]]) -> dict[str, Any]:
    """Same roll-up as provider_summary, but from plain (provider, fell_back)
    tuples so callers that persist dicts (not CallResult objects) can reuse it.
    """
    counts: dict[str, int] = {}
    fell_back = 0
    for serving, fb in serving_and_fellback:
        counts[serving] = counts.get(serving, 0) + 1
        if fb:
            fell_back += 1
    return {
        "calls": len(serving_and_fellback),
        "by_serving_provider": counts,
        "fallback_calls": fell_back,
        "mixed_provider_run": len(counts) > 1,
        "warning": (
            "Run served by multiple providers - per-case scores are not "
            "strictly comparable."
            if len(counts) > 1
            else None
        ),
    }
