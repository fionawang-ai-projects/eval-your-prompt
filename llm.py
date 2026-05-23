"""LLM client for Eval-Your-Prompt.

Exposes generate_eval(user_prompt) -> dict with keys:
    test_prompts: list[str]  — adversarial variations
    rubric:       str        — scoring rubric
    diagnosis:    str        — failure-mode analysis
"""

import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-opus-4-7"
MAX_TOKENS = 4096

SYSTEM_PROMPT = """You are an LLM prompt evaluator. Given a user's prompt, produce:
1. Ten adversarial test variations designed to expose weaknesses, ambiguities, or edge cases.
2. A scoring rubric (3-5 criteria) for judging an LLM's response to the original prompt.
3. A short diagnosis describing the most likely failure modes of the original prompt.

Always return your answer by calling the `submit_eval` tool."""

EVAL_TOOL = {
    "name": "submit_eval",
    "description": "Submit the structured evaluation of the user's prompt.",
    "input_schema": {
        "type": "object",
        "properties": {
            "test_prompts": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 10,
                "maxItems": 10,
                "description": "Exactly 10 adversarial test variations of the user's prompt.",
            },
            "rubric": {
                "type": "string",
                "description": "A 3-5 criterion scoring rubric for judging responses.",
            },
            "diagnosis": {
                "type": "string",
                "description": "Short analysis of the original prompt's likely failure modes.",
            },
        },
        "required": ["test_prompts", "rubric", "diagnosis"],
    },
}


def generate_eval(user_prompt: str) -> dict:
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=[EVAL_TOOL],
        tool_choice={"type": "tool", "name": "submit_eval"},
        messages=[{"role": "user", "content": user_prompt}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_eval":
            return block.input

    raise RuntimeError(f"Model did not call submit_eval. Got: {response.content!r}")


if __name__ == "__main__":
    sample = "Summarize this article in one sentence."
    result = generate_eval(sample)
    print(json.dumps(result, indent=2))
