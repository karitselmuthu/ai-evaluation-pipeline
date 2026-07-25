import os
import time
import asyncio
import logging
from typing import List, Dict, Any, Optional

from src.core.llm_client import GeneratorClient

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run a coroutine whether or not an event loop is already running
    (Streamlit runs inside one; the CLI does not)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
    return asyncio.run(coro)


class LLMGenerator:
    """
    Executes prompt generation against the LLM application being evaluated,
    routed through the local vault proxy (OpenAI-compatible). Fallback rotation
    is ALLOWED here: if the primary provider rate-limits, the proxy can let
    another serve the call so a long run stays alive. Every result records which
    provider actually served it (gen_serving_provider / gen_fell_back).

    Measures latency and token consumption.
    """

    DEFAULT_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the user question based on the provided context. If the context does not contain the answer, answer to the best of your knowledge but specify that it wasn't in the context.

Context:
{context}

Question: {question}
Answer:"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        provider: Optional[str] = None,
    ):
        # Provider is an attribution label; the proxy routes by the model string.
        # model "default" lets the proxy pick / rotate across providers.
        self.provider = provider or os.getenv("EVAL_GENERATOR_PROVIDER", "groq")
        self.model_name = model_name or os.getenv("EVAL_GENERATOR_MODEL", "default")
        self.temperature = temperature
        self.max_concurrency = int(os.getenv("EVAL_MAX_CONCURRENCY", "6"))
        self._client = GeneratorClient(provider=self.provider, model=self.model_name)

    async def generate_async(
        self,
        question: str,
        contexts: List[str],
        prompt_template: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Asynchronously generates an LLM answer for a question and records latency
        and provider attribution.
        """
        tpl = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        context_str = "\n---\n".join(contexts) if contexts else "No context provided."
        prompt_text = tpl.format(context=context_str, question=question)
        messages = [{"role": "user", "content": prompt_text}]

        start_time = time.perf_counter()
        serving_provider = self.provider
        fell_back = False
        input_tokens = len(prompt_text.split())
        output_tokens = 0

        try:
            # GeneratorClient is blocking (requests); run it off the event loop so
            # a batch of cases can be generated concurrently.
            result = await asyncio.to_thread(
                self._client.generate, messages, temperature=self.temperature
            )
            answer = result.text
            serving_provider = result.serving_provider
            fell_back = result.fell_back
            if result.prompt_tokens is not None:
                input_tokens = result.prompt_tokens
            output_tokens = (
                result.completion_tokens
                if result.completion_tokens is not None
                else len(answer.split())
            )
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            answer = f"Error during generation: {str(e)}"
            serving_provider = "error"

        latency = time.perf_counter() - start_time
        cost = 0.0  # Public/Internal API cost estimation

        return {
            "answer": answer,
            "latency": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "gen_requested_provider": self.provider,
            "gen_serving_provider": serving_provider,
            "gen_fell_back": fell_back,
        }

    def generate(
        self,
        question: str,
        contexts: List[str],
        prompt_template: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for generate_async."""
        return _run_async(self.generate_async(question, contexts, prompt_template))

    async def generate_batch_async(
        self,
        cases: List[Dict[str, Any]],
        prompt_template: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate over a batch of cases concurrently (bounded by
        EVAL_MAX_CONCURRENCY to stay under provider rate limits)."""
        sem = asyncio.Semaphore(self.max_concurrency)

        async def one(case: Dict[str, Any]) -> Dict[str, Any]:
            async with sem:
                gen_data = await self.generate_async(
                    question=case["question"],
                    contexts=case.get("contexts", []),
                    prompt_template=prompt_template,
                )
            return {**case, **gen_data}

        return await asyncio.gather(*[one(case) for case in cases])

    def generate_batch(
        self,
        cases: List[Dict[str, Any]],
        prompt_template: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper. Runs generation over the batch concurrently."""
        return _run_async(self.generate_batch_async(cases, prompt_template))
