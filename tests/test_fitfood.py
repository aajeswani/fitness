"""Contract tests for fitfood, generated from fitness/fitfood_python.prompt.

Every test names the rule id it enforces. The MUST NOT rules -- R2, R6, R8,
R11 -- are tested adversarially: the fixtures are built so the forbidden thing
fails loudly rather than merely going unobserved.

Nothing here touches the network. `recommend` takes its client as a parameter,
`build_request` is pure, and the one place that constructs a real client --
`main` -- is exercised with that constructor monkeypatched. Two rules are
checked against the module's own AST rather than its behaviour, because "never
inlines the model id" and "never reads ANTHROPIC_API_KEY" are properties of the
source that no call can demonstrate the absence of.
"""

import ast
import inspect
import io
import socket
import sys
from pathlib import Path

import anthropic
import httpx
import pytest
from pydantic import ValidationError

import fitfood
from fitfood import (
    DISCLAIMER,
    MODEL,
    MenuItem,
    Recommendations,
    build_request,
    is_plausible,
    main,
    recommend,
    render,
)

SOURCE = Path(fitfood.__file__).read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)

GOAL = "high protein low carb"
PLACE = "Chipotle"


# --- fixtures -----------------------------------------------------------

def item(**kw) -> MenuItem:
    base = dict(
        name="Chicken Burrito Bowl",
        calories=625,
        protein_g=45,
        carbs_g=40,
        why="45 g of protein against only 40 g of carbs.",
    )
    return MenuItem(**{**base, **kw})


def good(n: int = 3, **kw) -> Recommendations:
    """A recognized restaurant with `n` distinguishable items."""
    base = dict(
        recognized=True,
        restaurant=PLACE,
        items=[item(name=f"Item {i}", calories=300 + i) for i in range(n)],
    )
    return Recommendations(**{**base, **kw})


def unknown(name: str = "zzqx grill") -> Recommendations:
    return Recommendations(recognized=False, restaurant=name, items=[])


def placeholder() -> Recommendations:
    """The observed failure R12 exists to catch: valid, and obviously filler."""
    return Recommendations(
        recognized=True,
        restaurant=PLACE,
        items=[MenuItem(name="x", calories=0, protein_g=0, carbs_g=0, why="x")] * 3,
    )


class FakeResponse:
    def __init__(self, parsed, stop_reason="end_turn"):
        self._parsed = parsed
        self.stop_reason = stop_reason

    @property
    def parsed_output(self):
        return self._parsed


class RefusalResponse:
    """A refusal whose body is unreadable, as this model's refusals can be.

    R7 says the stop_reason check precedes the read. A read-first
    implementation raises AssertionError here instead of returning None, which
    is the whole point of making the property explode rather than return None.
    """

    stop_reason = "refusal"

    @property
    def parsed_output(self):
        raise AssertionError("R7: parsed_output read before the refusal check")

    @property
    def content(self):
        raise AssertionError("R7: content read before the refusal check")


class FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError(
                f"R12: more than {len(self.calls) - 1} calls; the module asked "
                "the API again after its budget was spent"
            )
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeClient:
    """Injected in place of anthropic.Anthropic. Never opens a socket."""

    def __init__(self, *responses):
        self.messages = FakeMessages(responses)

    @property
    def calls(self):
        return self.messages.calls


def api_error(cls, status):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {"error": {"type": "x", "message": "upstream said no"}}
    response = httpx.Response(status, request=request, json=body)
    return cls("upstream said no", response=response, body=body)


def run_main(monkeypatch, client, answers=(GOAL, PLACE)):
    """Drive main with injected I/O and a fake constructor. Returns (code, out)."""
    supplied = list(answers)
    shown = []

    def ask(_prompt):
        value = supplied.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(fitfood.anthropic, "Anthropic", lambda *a, **k: client)
    code = main(ask=ask, show=shown.append)
    return code, "\n".join(shown)


def function_node(name):
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"module defines no top-level {name}()")


def string_constants(node):
    return [
        n.value for n in ast.walk(node)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]


# --- R1: the model id is a constant, not a literal at the call site -----

def test_r1_request_targets_claude_opus_5():
    assert MODEL == "claude-opus-5"
    assert build_request(GOAL, PLACE)["model"] == "claude-opus-5"


def test_r1_call_site_references_the_constant_rather_than_inlining_it():
    """Negative, on the source: a test and the request must not disagree.

    An inlined literal lets someone change MODEL and ship a request that still
    goes somewhere else, with every behavioural test above still passing.
    """
    inlined = [s for s in string_constants(function_node("build_request"))
               if s == MODEL]
    assert not inlined, (
        "R1: build_request inlines the model id instead of reading MODEL; "
        "editing the constant would then not change the request"
    )


def test_r1_recommend_sends_exactly_what_build_request_returns():
    """Anchors every request-shape rule below to the real call.

    R2, R3, and R4 are asserted against build_request's return value. That is
    only meaningful if recommend passes it through untouched.
    """
    client = FakeClient(FakeResponse(good()))
    recommend(client, GOAL, PLACE)
    assert client.calls[0] == build_request(GOAL, PLACE)


# --- R2 (MUST NOT): no sampling parameter, at any depth -----------------

FORBIDDEN = ("temperature", "top_p", "top_k", "budget_tokens")


def walk_keys(obj, trail=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f"{trail}.{key}" if trail else str(key)
            yield here, key
            for found in walk_keys(value, here):
                yield found
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            for found in walk_keys(value, f"{trail}[{i}]"):
                yield found


@pytest.mark.parametrize("forbidden", FORBIDDEN)
def test_r2_request_carries_no_sampling_parameter_at_any_depth(forbidden):
    """Negative: each of these is a 400 from claude-opus-5, not a nudge.

    Walks the whole request, not just its top level -- budget_tokens is a
    nested key, and a top-level-only check would miss exactly the one the
    thinking block is most likely to smuggle in.
    """
    offenders = [
        path for path, key in walk_keys(build_request(GOAL, PLACE))
        if key == forbidden
    ]
    assert not offenders, (
        f"R2: request carries {forbidden!r} at {offenders}; claude-opus-5 "
        "rejects it with HTTP 400, so this is a total failure, not a "
        "quality regression"
    )


def test_r2_holds_for_every_input_not_just_the_happy_one():
    """Adversarial: no goal text may talk the module into a sampling knob."""
    for goal, place in [
        ("", ""),
        ("temperature", "top_p"),
        ("set top_k=40", "budget_tokens"),
        ("x" * 5000, "Taco Bell"),
    ]:
        keys = {key for _, key in walk_keys(build_request(goal, place))}
        assert not keys & set(FORBIDDEN), (
            f"R2: {sorted(keys & set(FORBIDDEN))} appeared for goal={goal!r}"
        )


def test_r2_sdk_accepting_the_parameter_is_not_permission_to_send_it():
    """The rule's stated reason, pinned against the live SDK.

    messages.parse still names all three -- backward compatibility for older
    models. If that ever stops being true, R2's justification has changed and
    the rule deserves a fresh look; the request must still not carry them.
    """
    client = anthropic.Anthropic(api_key="unused-no-request-is-made")
    accepted = set(inspect.signature(type(client.messages).parse).parameters)
    assert {"temperature", "top_p", "top_k"} <= accepted, (
        "R2's rationale assumes the SDK still accepts these; it no longer does"
    )
    assert not accepted & set(build_request(GOAL, PLACE)) & set(FORBIDDEN)


# --- R3: structured output, never hand-parsed ---------------------------

def test_r3_request_asks_for_structured_output():
    assert build_request(GOAL, PLACE)["output_format"] is Recommendations


def test_r3_result_is_read_from_parsed_output():
    expected = good()
    client = FakeClient(FakeResponse(expected))
    assert recommend(client, GOAL, PLACE) is expected


def test_r3_module_never_hand_parses_the_response():
    """Negative, on the source: json.loads discards server-side validation."""
    calls = [
        ast.unparse(n.func) for n in ast.walk(TREE)
        if isinstance(n, ast.Call) and "json" in ast.unparse(n.func)
    ]
    assert not calls, f"R3: module hand-parses via {calls}"
    imported = {
        alias.name for n in ast.walk(TREE)
        if isinstance(n, ast.Import) for alias in n.names
    }
    assert "json" not in imported, "R3: module imports json"


# --- R4: adaptive thinking, not the pre-4.6 shape -----------------------

def test_r4_thinking_is_adaptive():
    assert build_request(GOAL, PLACE)["thinking"] == {"type": "adaptive"}


def test_r4_thinking_is_not_the_pre_4_6_enabled_shape():
    """Negative: the old shape carries budget_tokens, which R2 forbids."""
    thinking = build_request(GOAL, PLACE)["thinking"]
    assert thinking.get("type") != "enabled", (
        "R4: {'type': 'enabled', 'budget_tokens': N} is the pre-4.6 shape and "
        "is rejected outright by this model"
    )
    assert set(thinking) == {"type"}


# --- R5: exactly three items, in the model's order ----------------------

def test_r5_recognized_restaurant_yields_three_items():
    recs = recommend(FakeClient(FakeResponse(good())), GOAL, PLACE)
    assert len(recs.items) == 3


def test_r5_model_ordering_is_preserved_exactly():
    """Negative: no re-sort, no re-score, no filter.

    The fixture is ordered so that sorting by calories, protein, or name would
    each produce a different sequence from the one the model returned.
    """
    ranked = Recommendations(
        recognized=True,
        restaurant=PLACE,
        items=[
            item(name="Zeta Bowl", calories=100, protein_g=50),
            item(name="Alpha Wrap", calories=900, protein_g=10),
            item(name="Mid Salad", calories=400, protein_g=30),
        ],
    )
    recs = recommend(FakeClient(FakeResponse(ranked)), GOAL, PLACE)
    assert [i.name for i in recs.items] == ["Zeta Bowl", "Alpha Wrap", "Mid Salad"]

    body = render(recs)
    assert body.index("Zeta Bowl") < body.index("Alpha Wrap") < body.index("Mid Salad")
    assert "1. Zeta Bowl" in body and "3. Mid Salad" in body


def test_r5_macros_are_integers():
    it = item()
    assert all(isinstance(v, int) for v in (it.calories, it.protein_g, it.carbs_g))
    with pytest.raises(ValidationError):
        item(calories=625.5)


def test_r5_recognized_is_required_not_defaulted():
    """A response omitting the flag must fail, not silently read False."""
    with pytest.raises(ValidationError):
        Recommendations(restaurant=PLACE, items=[])


def test_r5_render_shows_every_field_for_every_item():
    recs = good()
    body = render(recs)
    for it in recs.items:
        assert it.name in body
        assert str(it.calories) in body
        assert f"{it.protein_g}g protein" in body
        assert f"{it.carbs_g}g carbs" in body
        assert it.why in body


# --- R6 (MUST NOT): never invent a menu ---------------------------------

def test_r6_unknown_restaurant_renders_a_plain_message():
    """R6 pins the properties, not the sentence: the name is echoed back so the
    user can see whether a typo or the chain is why they got nothing, and no
    nutrition disclaimer appears when no nutrition figure was produced."""
    body = render(unknown())
    assert "zzqx grill" in body, "R6: message must echo the restaurant asked for"
    assert DISCLAIMER not in body, "R6: nothing was estimated, so nothing to disclaim"


@pytest.mark.parametrize(
    "recs",
    [unknown(), Recommendations(recognized=True, restaurant=PLACE, items=[])],
    ids=["unrecognized", "recognized-but-empty"],
)
def test_r6_no_item_line_is_ever_fabricated(recs):
    """Adversarial: both empty shapes must produce zero item lines.

    An invented burrito with invented macros is indistinguishable from a real
    one, which is what makes this the worst output the module could produce.
    A numbered line is the signature of one.
    """
    body = render(recs)
    numbered = [
        line for line in body.splitlines()
        if line[:1].isdigit() and line[1:3] == ". "
    ]
    assert not numbered, f"R6: fabricated item lines {numbered}"
    for token in ("cal", "protein", "carbs", "Why:"):
        assert token not in body, f"R6: rendered {token!r} with no items"


def test_r6_recommend_returns_the_empty_result_rather_than_filling_it_in():
    """Negative: the retry in R12 must not be repurposed into invention."""
    client = FakeClient(FakeResponse(unknown()))
    recs = recommend(client, GOAL, "zzqx grill")
    assert recs.items == []
    assert recs.recognized is False


def test_r6_unknown_restaurant_carries_no_disclaimer():
    """There are no estimates to disclaim, and R10 scopes itself to items."""
    assert DISCLAIMER not in render(unknown())


# --- R7: refusal is checked before the body is read ---------------------

def test_r7_refusal_returns_none():
    assert recommend(FakeClient(RefusalResponse()), GOAL, PLACE) is None


def test_r7_refusal_check_precedes_any_read_of_the_body():
    """Negative: reading first raises an opaque error instead of reporting.

    RefusalResponse.parsed_output raises. Returning None proves the read never
    happened; an AssertionError escaping here proves it did.
    """
    client = FakeClient(RefusalResponse())
    assert recommend(client, GOAL, PLACE) is None
    assert len(client.calls) == 1, "R7: a refusal must not be retried"


def test_r7_refusal_exits_non_zero_with_a_readable_message(monkeypatch):
    code, out = run_main(monkeypatch, FakeClient(RefusalResponse()))
    assert code == 1
    assert "declined" in out.lower()


# --- R8 (MUST NOT): no key handling anywhere ----------------------------

def test_r8_no_key_literal_appears_in_the_source():
    """Negative: the prefix must not occur even inside an error message.

    An example key in help text is the most common way one gets committed for
    real, so the rule bans the substring outright rather than the assignment.
    """
    assert "sk-ant-" not in SOURCE, (
        "R8: the source contains an 'sk-ant-' string"
    )


def test_r8_client_is_constructed_with_no_arguments(monkeypatch):
    """The SDK's own chain resolves credentials; passing one bypasses it."""
    seen = []

    class Recorder:
        def __init__(self, *args, **kwargs):
            seen.append((args, kwargs))
            self.messages = FakeMessages([FakeResponse(good())])

    monkeypatch.setattr(fitfood.anthropic, "Anthropic", Recorder)
    supplied = [GOAL, PLACE]
    assert main(ask=lambda _p: supplied.pop(0), show=lambda _t: None) == 0
    assert seen == [((), {})], f"R8: client constructed with {seen}"


def test_r8_module_never_reads_the_environment():
    """Negative, on the source: an unset key does not mean no credentials.

    Branching on os.environ['ANTHROPIC_API_KEY'] refuses to run for every user
    whose credentials come from `ant auth login`, a workload identity, or the
    on-disk profile -- all of which the SDK resolves on its own.
    """
    reads = [
        ast.unparse(node) for node in ast.walk(TREE)
        if isinstance(node, ast.Attribute) and node.attr in {"environ", "getenv"}
    ]
    assert not reads, f"R8: module reads the environment via {reads}"
    assert "os.environ" not in SOURCE and "getenv" not in SOURCE


def test_r8_no_credential_file_is_opened():
    """The non-responsibility behind the rule: no key is ever written out."""
    writes = [
        ast.unparse(node) for node in ast.walk(TREE)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) in {"open", "Path", "write_text"}
    ]
    assert not writes, f"R8: module performs file I/O via {writes}"


# --- R9: five failures, five remedies -----------------------------------

ERRORS = [
    ("auth", api_error(anthropic.AuthenticationError, 401), "credentials"),
    ("not_found", api_error(anthropic.NotFoundError, 404), MODEL),
    ("rate_limit", api_error(anthropic.RateLimitError, 429), "rate limit"),
    ("status", api_error(anthropic.APIStatusError, 500), "500"),
    (
        "connection",
        anthropic.APIConnectionError(
            message="down",
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        ),
        "network",
    ),
]


@pytest.mark.parametrize(
    "error,expected", [(e, t) for _, e, t in ERRORS],
    ids=[name for name, _, _ in ERRORS],
)
def test_r9_each_api_failure_gets_its_own_message(monkeypatch, capsys, error, expected):
    code, _ = run_main(monkeypatch, FakeClient(error))
    assert code != 0
    err = capsys.readouterr().err
    assert expected.lower() in err.lower(), (
        f"R9: {type(error).__name__} produced {err.strip()!r}, which does not "
        f"mention {expected!r}"
    )


def test_r9_the_five_messages_are_distinct(monkeypatch, capsys):
    """Negative: a single `except Exception` collapses them into one string.

    Collapsing tells a user with an expired token to check their network. Five
    remedies must reach the user as five messages.
    """
    messages = []
    for _, error, _ in ERRORS:
        run_main(monkeypatch, FakeClient(error))
        messages.append(capsys.readouterr().err.strip())
    assert len(set(messages)) == 5, (
        "R9: expected five distinct stderr messages, got "
        f"{len(set(messages))}: {sorted(set(messages))}"
    )


def test_r9_specific_status_errors_are_not_swallowed_by_the_supertype(
    monkeypatch, capsys
):
    """Adversarial, on the ordering: APIStatusError first eats the three.

    AuthenticationError, NotFoundError, and RateLimitError are all subclasses
    of APIStatusError. Handlers written in the wrong order still catch every
    error -- they just report all of them as the generic one. Comparing each
    specific message against the generic one is what detects that.
    """
    run_main(monkeypatch, FakeClient(api_error(anthropic.APIStatusError, 500)))
    generic = capsys.readouterr().err.strip()

    for name, error, _ in ERRORS[:3]:
        run_main(monkeypatch, FakeClient(error))
        specific = capsys.readouterr().err.strip()
        assert specific != generic, (
            f"R9: {name} reported as the generic APIStatusError message "
            f"{generic!r}; the handler order lets the supertype swallow it"
        )


def test_r9_errors_go_to_stderr_not_to_the_output_channel(monkeypatch, capsys):
    code, out = run_main(monkeypatch, FakeClient(ERRORS[0][1]))
    assert code == 1
    assert out == ""
    assert capsys.readouterr().err.strip()


# --- R10: the estimate disclaimer -----------------------------------

def test_r10_disclaimer_says_the_figures_are_estimates():
    assert "estimate" in DISCLAIMER.lower()


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_r10_every_result_with_items_carries_the_disclaimer(n):
    assert DISCLAIMER in render(good(n))


def test_r10_disclaimer_survives_the_full_path(monkeypatch):
    code, out = run_main(monkeypatch, FakeClient(FakeResponse(good())))
    assert code == 0
    assert DISCLAIMER in out


def test_r10_disclaimer_is_one_shared_constant():
    """Negative: a re-typed sentence drifts from the one tests assert on.

    Counted over AST string constants, not raw source text: the definition may
    wrap across lines, and adjacent literals fold into a single node.
    """
    literals = [
        node.value
        for node in ast.walk(ast.parse(SOURCE))
        if isinstance(node, ast.Constant) and node.value == DISCLAIMER
    ]
    assert len(literals) == 1, "R10: disclaimer text appears outside the constant"


# --- R11 (MUST NOT): the pure three stay pure ---------------------------

class Tripwire(io.StringIO):
    def write(self, text):
        if text.strip():
            raise AssertionError(f"R11: wrote to a stream: {text!r}")
        return len(text)


@pytest.mark.parametrize(
    "call",
    [
        lambda: build_request(GOAL, PLACE),
        lambda: is_plausible(good()),
        lambda: render(good()),
        lambda: render(unknown()),
    ],
    ids=["build_request", "is_plausible", "render", "render-unknown"],
)
def test_r11_pure_functions_perform_no_io(monkeypatch, call):
    """Adversarial: stdout, stderr, and the socket layer all made to explode.

    Every rule above must be checkable without a network round trip. A print
    or a connect inside these three would make that false, so the fixture
    turns each into a failure rather than trusting the reading.
    """
    def no_sockets(*a, **k):
        raise AssertionError("R11: opened a socket")

    monkeypatch.setattr(socket, "socket", no_sockets)
    monkeypatch.setattr(socket, "create_connection", no_sockets)
    monkeypatch.setattr(sys, "stdout", Tripwire())
    monkeypatch.setattr(sys, "stderr", Tripwire())
    call()


@pytest.mark.parametrize(
    "call",
    [lambda r: is_plausible(r), lambda r: render(r)],
    ids=["is_plausible", "render"],
)
def test_r11_pure_functions_do_not_mutate_their_argument(call):
    """Negative: a re-sort in place would satisfy R5's return value and lie."""
    recs = good()
    before = recs.model_dump()
    call(recs)
    assert recs.model_dump() == before, "R11: argument was mutated"


def test_r11_only_main_constructs_a_client():
    """Negative, on the source: a client built elsewhere is unfakeable."""
    for name in ("build_request", "is_plausible", "render", "recommend"):
        node = function_node(name)
        constructors = [
            ast.unparse(n) for n in ast.walk(node)
            if isinstance(n, ast.Call) and "Anthropic" in ast.unparse(n.func)
        ]
        assert not constructors, (
            f"R11: {name} constructs a client ({constructors}); tests can no "
            "longer inject a fake and every rule above needs the network"
        )


def test_r11_render_returns_text_rather_than_printing():
    body = render(good())
    assert isinstance(body, str) and body.strip()


# --- R12: the placeholder response, at its exact boundaries -------------

def test_r12_a_real_result_is_plausible():
    assert is_plausible(good()) is True


def test_r12_the_observed_placeholder_is_rejected():
    assert is_plausible(placeholder()) is False


@pytest.mark.parametrize(
    "calories,expected",
    [(-1, False), (0, False), (1, True), (625, True)],
)
def test_r12_calorie_boundary_sits_at_zero(calories, expected):
    """Boundary: the rule says non-positive, so zero fails and one passes."""
    recs = good()
    recs.items[1] = item(calories=calories)
    assert is_plausible(recs) is expected


@pytest.mark.parametrize(
    "length,expected", [(0, False), (1, False), (2, False), (3, True), (20, True)]
)
def test_r12_name_boundary_sits_at_two_characters(length, expected):
    """Boundary: two or fewer fails, three passes."""
    recs = good()
    recs.items[0] = item(name="n" * length)
    assert is_plausible(recs) is expected


@pytest.mark.parametrize(
    "length,expected", [(0, False), (14, False), (15, False), (16, True), (60, True)]
)
def test_r12_why_boundary_sits_at_fifteen_characters(length, expected):
    """Boundary: fifteen or fewer fails, sixteen passes."""
    recs = good()
    recs.items[2] = item(why="w" * length)
    assert is_plausible(recs) is expected


@pytest.mark.parametrize(
    "n,expected", [(0, False), (1, False), (2, False), (3, True), (4, False)]
)
def test_r12_recognized_result_must_carry_exactly_three_items(n, expected):
    assert is_plausible(good(n)) is expected


@pytest.mark.parametrize("n", [0, 1, 3])
def test_r12_an_unrecognized_result_is_never_implausible(n):
    """Negative: the shape check must not fire on R6's correct empty answer.

    An implementation that runs the item-count test before the recognized test
    calls every unknown restaurant a placeholder and burns a retry on it.
    """
    recs = unknown()
    recs.items = good(n).items if n else []
    assert is_plausible(recs) is True, (
        "R12: unrecognized result judged implausible; R6 defines its empty "
        "list as the correct answer"
    )


def test_r12_a_placeholder_is_retried_exactly_once():
    real = good()
    client = FakeClient(FakeResponse(placeholder()), FakeResponse(real))
    assert recommend(client, GOAL, PLACE) is real
    assert len(client.calls) == 2


def test_r12_a_plausible_first_answer_is_not_retried():
    """Negative: an unconditional second call doubles cost and latency."""
    client = FakeClient(FakeResponse(good()))
    recommend(client, GOAL, PLACE)
    assert len(client.calls) == 1


def test_r12_an_unrecognized_first_answer_is_not_retried():
    client = FakeClient(FakeResponse(unknown()))
    recommend(client, GOAL, "zzqx grill")
    assert len(client.calls) == 1


def test_r12_two_placeholders_return_the_second_rather_than_raising():
    """The rule's stated tradeoff: a degraded answer beats an exception.

    FakeMessages raises on a third call, so a loop that keeps retrying fails
    here rather than spinning.
    """
    second = placeholder()
    client = FakeClient(FakeResponse(placeholder()), FakeResponse(second))
    assert recommend(client, GOAL, PLACE) is second
    assert len(client.calls) == 2


def test_r12_never_exceeds_two_calls_for_any_response_sequence():
    """Adversarial: the third call is a hard error, not a slow path."""
    for first in (placeholder(), good(2), good(4)):
        client = FakeClient(FakeResponse(first), FakeResponse(placeholder()))
        recommend(client, GOAL, PLACE)
        assert len(client.calls) <= 2


def test_r12_the_retried_call_is_identical_to_the_first():
    """A retry that quietly reshapes the request would evade R1-R4 entirely."""
    client = FakeClient(FakeResponse(placeholder()), FakeResponse(good()))
    recommend(client, GOAL, PLACE)
    assert client.calls[0] == client.calls[1] == build_request(GOAL, PLACE)


# --- R13: blank input never reaches the API -----------------------------

@pytest.mark.parametrize(
    "answers",
    [("", PLACE), (GOAL, ""), ("", ""), ("   ", PLACE), (GOAL, "\t\n "), ("  ", " ")],
    ids=["no-goal", "no-place", "neither", "spaces-goal", "tabs-place", "both-blank"],
)
def test_r13_blank_input_exits_non_zero(monkeypatch, capsys, answers):
    """Boundary: whitespace-only is blank, compared after stripping."""
    code, _ = run_main(monkeypatch, FakeClient(), answers=answers)
    assert code == 1
    assert capsys.readouterr().err.strip()


def test_r13_blank_input_makes_no_api_call(monkeypatch, capsys):
    """Adversarial: the constructor itself is a tripwire.

    A blank goal must be rejected before any client exists, so building one is
    made an outright failure rather than a wasted call nobody notices.
    """
    def forbidden(*a, **k):
        raise AssertionError("R13: built a client for a blank input")

    monkeypatch.setattr(fitfood.anthropic, "Anthropic", forbidden)
    supplied = ["   ", PLACE]
    assert main(ask=lambda _p: supplied.pop(0), show=lambda _t: None) == 1
    capsys.readouterr()


def test_r13_surrounding_whitespace_is_stripped_from_valid_input(monkeypatch):
    """The other half of "compared after stripping": padding is not content."""
    client = FakeClient(FakeResponse(good()))
    code, _ = run_main(monkeypatch, client, answers=(f"  {GOAL} ", f"\t{PLACE}\n"))
    assert code == 0
    assert client.calls[0] == build_request(GOAL, PLACE)


# --- R14: an interrupt is not an error ----------------------------------

@pytest.mark.parametrize("exc", [KeyboardInterrupt, EOFError])
@pytest.mark.parametrize("position", [0, 1], ids=["first-prompt", "second-prompt"])
def test_r14_interrupt_exits_130(monkeypatch, capsys, exc, position):
    answers = [GOAL, PLACE]
    answers[position] = exc()
    code, _ = run_main(monkeypatch, FakeClient(), answers=tuple(answers))
    assert code == 130, f"R14: {exc.__name__} at prompt {position} exited {code}"
    capsys.readouterr()


@pytest.mark.parametrize("exc", [KeyboardInterrupt, EOFError])
def test_r14_interrupt_prints_nothing_to_stderr(monkeypatch, capsys, exc):
    """Negative: an abort the user chose must not be reported as a failure."""
    code, out = run_main(monkeypatch, FakeClient(), answers=(exc(), PLACE))
    assert code == 130
    assert capsys.readouterr().err == "", "R14: stderr must stay empty"
    assert out == ""


def test_r14_interrupt_makes_no_api_call(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("R14: built a client after an interrupt")

    monkeypatch.setattr(fitfood.anthropic, "Anthropic", forbidden)

    def ask(_prompt):
        raise KeyboardInterrupt

    assert main(ask=ask, show=lambda _t: None) == 130


def test_r14_exit_codes_are_distinct_across_the_three_outcomes(monkeypatch):
    """130 must not collide with the 1 that R13 and R9 return."""
    ok, _ = run_main(monkeypatch, FakeClient(FakeResponse(good())))
    blank, _ = run_main(monkeypatch, FakeClient(), answers=("", ""))
    aborted, _ = run_main(monkeypatch, FakeClient(), answers=(EOFError(), PLACE))
    assert (ok, blank, aborted) == (0, 1, 130)
