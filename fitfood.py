"""Rank a fast food menu against a health goal.

Generated from fitness/fitfood_python.prompt. The module is `fitfood`, not
`fitness_app` -- see the naming note in that prompt.

Claude supplies both the menu and the ranking from its own knowledge, so the
nutrition figures are estimates. R10 requires every rendered result to say so.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict, List, Optional

import anthropic
from pydantic import BaseModel, Field

__all__ = [
    "MODEL",
    "MAX_TOKENS",
    "DISCLAIMER",
    "SYSTEM_PROMPT",
    "MenuItem",
    "Recommendations",
    "build_request",
    "recommend",
    "is_plausible",
    "render",
    "main",
]

MODEL = "claude-opus-5"
MAX_TOKENS = 16000

DISCLAIMER = (
    "Nutrition figures are estimates — check the restaurant for exact values."
)

# Concision and scope are stated explicitly: this model is verbose by default
# and will expand a ranking task into meal planning. No self-check instruction
# appears here -- it self-verifies, and asking again provokes over-verification.
SYSTEM_PROMPT = """You are a nutrition-savvy fast food guide.

Given a health goal and a restaurant, pick the 3 items from that restaurant's \
actual menu that best serve the goal, best first. Use standard portions and no \
modifications unless a modification is the whole point of the pick (say so in \
the item name if it is).

For each item give calories, protein in grams, carbs in grams, and one sentence \
on why it fits this specific goal. Cite the numbers that justify the pick rather \
than describing the food. Be concise; no preamble, no closing advice.

Nutrition values are your best estimate of the chain's published figures. If you \
do not recognize the restaurant, set recognized to false and return no items.

Rank only. Do not suggest other restaurants, build meal plans, or add caveats \
beyond the fields requested."""


class MenuItem(BaseModel):
    """One ranked menu item. Field descriptions are prompt, not documentation."""

    name: str = Field(description="Menu item name as the restaurant lists it")
    calories: int = Field(description="Calories for a standard portion")
    protein_g: int = Field(description="Grams of protein")
    carbs_g: int = Field(description="Grams of carbohydrate")
    why: str = Field(description="One sentence tying this item to the stated goal")


class Recommendations(BaseModel):
    """The full ranking for one restaurant."""

    recognized: bool = Field(
        description="False if the restaurant is unknown or not a food chain"
    )
    restaurant: str = Field(description="Canonical name of the restaurant")
    items: List[MenuItem] = Field(description="Top 3 items, best match first")


def build_request(goal: str, restaurant: str) -> Dict[str, Any]:
    """Build the exact kwargs for ``client.messages.parse``.

    Pure by contract (R11) so the request can be inspected without a network
    call. Carries no sampling parameter: R2 forbids temperature, top_p, top_k,
    and thinking.budget_tokens, each of which this model rejects with a 400.

    Args:
        goal: The user's stated health goal, free-form.
        restaurant: The restaurant name, free-form.

    Returns:
        Keyword arguments to splat into ``messages.parse``.
    """
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "thinking": {"type": "adaptive"},
        "messages": [
            {
                "role": "user",
                "content": f"Health goal: {goal}\nRestaurant: {restaurant}",
            }
        ],
        "output_format": Recommendations,
    }


def is_plausible(recs: Recommendations) -> bool:
    """Report whether a schema-valid result is also shaped like real data.

    The placeholder response -- three items named "x" with zero calories -- is
    observed behaviour, not a hypothetical, and it satisfies the schema. An
    unrecognized result is always plausible: R6 defines its empty list as the
    correct answer, so this check must not fire on it.

    Args:
        recs: A validated result to inspect.

    Returns:
        True if the result may be shown to the user.
    """
    if not recs.recognized:
        return True
    if len(recs.items) != 3:
        return False
    return all(
        item.calories > 0 and len(item.name) > 2 and len(item.why) > 15
        for item in recs.items
    )


def recommend(
    client: Any, goal: str, restaurant: str
) -> Optional[Recommendations]:
    """Ask Claude to rank the restaurant's menu against the goal.

    Makes at most two calls (R12): a placeholder response is retried once, and
    a second implausible result is returned rather than raised, since a
    degraded answer the user can see beats an exception they cannot act on.

    Args:
        client: An Anthropic client. Never constructed here, so tests inject.
        goal: The user's stated health goal.
        restaurant: The restaurant to rank.

    Returns:
        The ranking, or None if the model refused.
    """
    recs: Optional[Recommendations] = None

    for _ in range(2):
        response = client.messages.parse(**build_request(goal, restaurant))

        # R7: a refusal may carry no content at all, so this precedes any read.
        if response.stop_reason == "refusal":
            return None

        recs = response.parsed_output
        if recs is not None and is_plausible(recs):
            return recs

    return recs


def render(recs: Recommendations) -> str:
    """Render a ranking as user-facing text.

    Pure by contract (R11): returns the text and prints nothing.

    Args:
        recs: The ranking to render.

    Returns:
        The full text, including the R10 disclaimer when items are present.
    """
    if not recs.recognized or not recs.items:
        return f"Don't know the menu for {recs.restaurant!r}. Try a chain name."

    lines = [f"Top picks at {recs.restaurant}:", ""]
    for i, item in enumerate(recs.items, 1):
        lines.append(f"{i}. {item.name}  {item.calories} cal")
        lines.append(f"   {item.protein_g}g protein · {item.carbs_g}g carbs")
        lines.append(f"   Why: {item.why}")
        lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def main(
    ask: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
) -> int:
    """Collect input, rank, and render.

    The only symbol that constructs a client or touches the terminal.

    Args:
        ask: Prompt-and-read callable. Injected in tests.
        show: Output callable. Injected in tests.

    Returns:
        0 on success; 1 on blank input, refusal, or API failure; 130 on
        interrupt.
    """
    try:
        goal = ask("Health goal: ").strip()
        restaurant = ask("Restaurant: ").strip()
    except (EOFError, KeyboardInterrupt):
        # R14: an abort the user chose is not an error, so nothing on stderr.
        return 130

    if not goal or not restaurant:
        print("Need both a health goal and a restaurant.", file=sys.stderr)
        return 1

    # R8: bare constructor. The SDK's own chain resolves credentials; an unset
    # ANTHROPIC_API_KEY does not mean the user has none.
    try:
        client = anthropic.Anthropic()
    except Exception:
        print(
            "No Anthropic credentials found. Run `ant auth login`, or export "
            "ANTHROPIC_API_KEY=...",
            file=sys.stderr,
        )
        return 1

    # R9: most-specific first. APIStatusError is a supertype of the three
    # status errors above it and would otherwise swallow them.
    try:
        recs = recommend(client, goal, restaurant)
    except anthropic.AuthenticationError:
        print(
            "Credentials were rejected. Run `ant auth login`, or check "
            "ANTHROPIC_API_KEY.",
            file=sys.stderr,
        )
        return 1
    except anthropic.NotFoundError:
        print(f"Model {MODEL} is not available on this account.", file=sys.stderr)
        return 1
    except anthropic.RateLimitError:
        print("Rate limited by the API. Wait a moment and try again.", file=sys.stderr)
        return 1
    except anthropic.APIStatusError as e:
        print(f"API error {e.status_code}: {e.message}", file=sys.stderr)
        return 1
    except anthropic.APIConnectionError:
        print("Could not reach the API. Check your network.", file=sys.stderr)
        return 1

    if recs is None:
        show("Claude declined to answer that one. Try rephrasing the goal.")
        return 1

    show(render(recs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
