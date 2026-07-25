"""Secret access seam for the AI Evaluation Pipeline.

All secret reads go through SecretProvider. Nothing in the pipeline calls
os.getenv("GEMINI_API_KEY") or a cloud secret SDK directly.

Backend is chosen by the SECRET_BACKEND env var:
    SECRET_BACKEND=vault   -> read from the local LLM Key Vault daemon
    SECRET_BACKEND=env     -> read from environment / .env  (DEFAULT)

Default is "env" deliberately: CI has no local vault, and the CI gatekeeper
eval must keep working. Opt in to the vault locally.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

import requests


class SecretNotFound(RuntimeError):
    """Raised when a provider's key cannot be resolved from the backend."""


class SecretProvider(ABC):
    """The seam. Swap implementations; callers never change."""

    @abstractmethod
    def get_secret(self, provider: str) -> str:
        """Return the API key for `provider` (e.g. 'gemini', 'groq')."""

    def health(self) -> bool:
        """Cheap check that this backend is usable. Override where meaningful."""
        return True


class EnvSecretsProvider(SecretProvider):
    """Reads keys from environment variables / .env. Used in CI."""

    # provider name -> env var holding its key
    ENV_KEYS = {
        "gemini": "GEMINI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }

    def get_secret(self, provider: str) -> str:
        var = self.ENV_KEYS.get(provider, f"{provider.upper()}_API_KEY")
        val = os.getenv(var)
        if not val:
            raise SecretNotFound(
                f"No key for '{provider}': env var {var} is unset. "
                f"Set it, or run with SECRET_BACKEND=vault."
            )
        return val


class VaultSecretsProvider(SecretProvider):
    """Reads keys from the local OS-keychain-backed LLM Key Vault daemon.

    Keys never touch disk in plaintext: the daemon holds them in the OS
    keychain and authorizes callers with a session token written at launch.
    """

    def __init__(
        self,
        base_url: str | None = None,
        token_path: str | None = None,
        timeout: float = 5.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("VAULT_URL", "http://127.0.0.1:9876")).rstrip("/")
        self.token_path = Path(token_path or os.path.expanduser("~/.llm_vault_token"))
        self.timeout = timeout
        self._token: str | None = None

    def _read_token(self) -> str:
        """Load the session token written by the vault daemon at startup."""
        if self._token:
            return self._token
        try:
            self._token = self.token_path.read_text().strip()
        except FileNotFoundError as exc:
            raise SecretNotFound(
                f"Vault token not found at {self.token_path}. "
                f"Is the vault daemon running?"
            ) from exc
        if not self._token:
            raise SecretNotFound(f"Vault token at {self.token_path} is empty.")
        return self._token

    def health(self) -> bool:
        """True if the daemon is reachable and the token file exists."""
        try:
            self._read_token()
        except SecretNotFound:
            return False
        try:
            requests.get(self.base_url, timeout=self.timeout)
            return True
        except requests.RequestException:
            return False

    def get_secret(self, provider: str) -> str:
        token = self._read_token()
        url = f"{self.base_url}/api/v1/keys/{provider}"
        try:
            resp = requests.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise SecretNotFound(
                f"Vault daemon unreachable at {self.base_url}: {exc}"
            ) from exc

        if resp.status_code == 404:
            raise SecretNotFound(f"Vault has no key stored for provider '{provider}'.")
        if resp.status_code in (401, 403):
            raise SecretNotFound(
                f"Vault rejected the session token (HTTP {resp.status_code}). "
                f"Restart the daemon to regenerate {self.token_path}."
            )
        resp.raise_for_status()

        key = resp.json().get("api_key")
        if not key:
            raise SecretNotFound(f"Vault returned no api_key for '{provider}'.")
        return key


def get_secret_provider() -> SecretProvider:
    """Factory: pick the backend from SECRET_BACKEND (default: env).

    Fails loudly rather than silently degrading: if SECRET_BACKEND=vault is
    requested but the daemon is not reachable, that is an error worth seeing,
    not a quiet fallback to env that changes which key you're testing with.
    """
    backend = os.getenv("SECRET_BACKEND", "env").strip().lower()

    if backend == "vault":
        provider = VaultSecretsProvider()
        if not provider.health():
            raise SecretNotFound(
                "SECRET_BACKEND=vault but the vault daemon is not reachable. "
                "Start it, or set SECRET_BACKEND=env."
            )
        return provider

    if backend == "env":
        return EnvSecretsProvider()

    raise ValueError(f"Unknown SECRET_BACKEND '{backend}'. Use 'vault' or 'env'.")
