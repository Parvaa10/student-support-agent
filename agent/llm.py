"""Thin wrapper around the Google Gemini API (google-genai SDK).

* Automatic function calling is DISABLED on purpose: the agent loop in agent.py executes tools itself,
  so every decision, tool call, latency and token count can be traced.
* Retries with exponential back-off on rate limits (429) and transient server errors (5xx).
"""
import time

from google import genai
from google.genai import errors, types

from . import config
from .tools import function_declarations


class GeminiLLM:
    def __init__(self, api_key: str | None = config.GEMINI_API_KEY, model: str = config.GEMINI_MODEL):
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.tool = types.Tool(function_declarations=[types.FunctionDeclaration(**d) for d in function_declarations()])

    def generate(self, contents: list, system_instruction: str, max_retries: int = 4):
        cfg = dict(
            system_instruction=system_instruction,
            tools=[self.tool],
            temperature=0.1,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        if config.GEMINI_THINKING_BUDGET is not None:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=config.GEMINI_THINKING_BUDGET)
        delay = 4.0
        for attempt in range(max_retries + 1):
            try:
                return self.client.models.generate_content(
                    model=self.model, contents=contents, config=types.GenerateContentConfig(**cfg))
            except errors.APIError as e:
                if getattr(e, "code", None) in (429, 500, 502, 503, 504) and attempt < max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise


def usage(resp) -> dict:
    u = getattr(resp, "usage_metadata", None)
    if not u:
        return {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "total_tokens": 0}
    return {
        "input_tokens": u.prompt_token_count or 0,
        "output_tokens": u.candidates_token_count or 0,
        "thinking_tokens": getattr(u, "thoughts_token_count", 0) or 0,
        "total_tokens": u.total_token_count or 0,
    }
