"""ChartQA relaxed scoring, with explicitly versioned edge-case semantics.

Reference: https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py
Independent implementation of the metric convention; no external judge.
"""

from dataclasses import asdict, dataclass
import math

from .protocol import parse_action


SCORING_PROTOCOL = "chartqa_pix2struct_trim_finite_v1"


def _number(value: str) -> float | None:
    try:
        number = float(value[:-1]) / 100 if value.endswith("%") else float(value)
        return number if math.isfinite(number) else None
    except (ValueError, OverflowError):
        return None


def relaxed_correctness(target: str, prediction: str) -> bool:
    """5% relative numeric tolerance; case-insensitive text matching otherwise.

    Strip outside whitespace, preserve internal whitespace/punctuation/units.
    Percentages convert to fractions. Zero targets use textual matching, as in
    the Pix2Struct reference (therefore '0' != '0.0'). Non-finite numeric spellings
    never count as valid answers, even if textually identical. No comma removal,
    currency conversion, ANLS, or year-specific exceptions are used.
    """
    target, prediction = target.strip(), prediction.strip()
    if not target or not prediction:
        return False
    for text in (target, prediction):
        try:
            value = float(text[:-1] if text.endswith("%") else text)
        except (ValueError, OverflowError):
            continue
        if not math.isfinite(value):
            return False
    gold, pred = _number(target), _number(prediction)
    if gold is not None and gold != 0 and pred is not None:
        return abs(pred - gold) / abs(gold) <= 0.05
    return target.lower() == prediction.lower()


@dataclass(frozen=True)
class Score:
    protocol: str
    parsed: bool
    prediction: str | None
    relaxed: bool
    strict: bool
    reward: float
    error: str | None

    def to_dict(self):
        return asdict(self)


def score_completion(completion: str, target: str) -> Score:
    if not isinstance(target, str) or not target.strip():
        raise ValueError("Expected one non-empty ChartQA target string")
    action = parse_action(completion)
    if action.kind != "answer":
        return Score(SCORING_PROTOCOL, False, None, False, False, 0.0, action.error or "no_final_answer")
    prediction = action.answer
    correct = relaxed_correctness(target, prediction)
    return Score(SCORING_PROTOCOL, True, prediction, correct, target.strip() == prediction, float(correct), None)
