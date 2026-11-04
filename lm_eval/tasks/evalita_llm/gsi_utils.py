"""Generative GSI:detect regression and category classification.

The gold mappings come from evalitahf/gsi_detect. Predictions are continuous
values or one canonical category, not multiple-choice letters. Metrics follow
the organizer evaluation script supplied for GSI:detect: invalid values become
0, invalid categories become an empty string, and F1 uses observed labels.
Official task: https://gsi-d-evalita.fbk.eu/task-definition
"""

import json
import math
import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation


def _final_answer(value):
    if isinstance(value, Mapping):
        # Never promote raw_output or reasoning if a final response is missing.
        value = value.get("final_answer", "")
    if not isinstance(value, str):
        return ""
    text = value.strip()
    # GPT-OSS/Harmony style explicit final channel.
    final_marker = re.compile(r"<\|channel\|>final(?:<\|message\|>|\s*)")
    matches = list(final_marker.finditer(text))
    if matches:
        text = text[matches[-1].end() :]
    elif "<|channel|>" in text or "<|message|>" in text:
        return ""
    # Gemma's native thinking field closes with <channel|>.
    while "<|channel>thought\n" in text:
        start = text.index("<|channel>thought\n")
        end = text.find("<channel|>", start)
        if end < 0:
            return ""
        text = text[:start] + text[end + len("<channel|>") :]
    if "<|channel>" in text or "<channel|>" in text:
        return ""
    for tag in ["think", "analysis"]:
        opening = re.compile(r"<\s*" + tag + r"\s*>", re.IGNORECASE)
        closing = re.compile(r"<\s*/\s*" + tag + r"\s*>", re.IGNORECASE)
        text = re.sub(
            r"<\s*" + tag + r"\s*>.*?<\s*/\s*" + tag + r"\s*>",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if opening.search(text):
            return ""
        endings = list(closing.finditer(text))
        if endings:
            text = text[endings[-1].end() :]
    text = re.sub(
        r"^(?:<\|im_start\|>assistant\s*|<｜Assistant｜>|<\|assistant\|>)", "", text
    )
    for marker in [
        "</s>",
        "<|im_end|>",
        "<|endoftext|>",
        "<｜end▁of▁sentence｜>",
        "<|eot_id|>",
        "<turn|>",
        "<|return|>",
        "<|end|>",
    ]:
        text = text.split(marker, 1)[0]
    if any(
        marker in text
        for marker in ["<|im_start|>", "<|start|>", "<|channel|>", "<|user|>"]
    ):
        return ""
    return text.strip()


def _first_result(results):
    if isinstance(results, (str, bytes)):
        raise TypeError(
            "results must be the harness sequence of generations, not a string"
        )
    return _final_answer(results[0]) if results else ""


# GSI parsing, targets and corpus-level metric aggregators.
CATEGORIES = ("role", "personality", "competence", "physical", "sexual", "relational")


NUMBER = r"[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][+-]?\d+)?"


VALUE_CUE = re.compile(
    r"(?:\b(?:gs_value|risposta|answer|valore|score)\b)\s*(?:finale\s*)?(?::|=|è|is)\s*",
    re.IGNORECASE,
)


CATEGORY_CUE = re.compile(
    r"(?:\b(?:gs_category|categoria|category|risposta|answer)\b)\s*(?:finale\s*)?(?::|=|è|is)\s*",
    re.IGNORECASE,
)


def _unwrap(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(
            r"^```(?:json|text)?\s*\n?", "", text, count=1, flags=re.IGNORECASE
        )
        text = text[:-3].strip()
    pairs = [
        ("**", "**"),
        ("__", "__"),
        ("`", "`"),
        ('"', '"'),
        ("'", "'"),
        ("“", "”"),
        ("*", "*"),
    ]
    changed = True
    while changed:
        changed = False
        for left, right in pairs:
            if (
                len(text) > len(left) + len(right)
                and text.startswith(left)
                and text.endswith(right)
            ):
                text = text[len(left) : -len(right)].strip()
                changed = True
                break
    return text


def _value(text):
    if isinstance(text, bool):
        return None
    text = _unwrap(str(text))
    if (
        text.endswith(".")
        and not re.fullmatch(NUMBER, text)
        and re.fullmatch(NUMBER, text[:-1])
    ):
        text = text[:-1]
    if not re.fullmatch(NUMBER, text):
        return None
    try:
        value = Decimal(text.replace(",", "."))
    except (InvalidOperation, ValueError):
        return None
    return float(value) if value.is_finite() and 0 <= value <= 1 else None


def _json_field(text, field):
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return False, None
    return (
        (True, obj[field]) if isinstance(obj, dict) and field in obj else (False, None)
    )


def parse_gsi_value(response):
    text = _unwrap(_final_answer(response))
    direct = _value(text)
    if direct is not None:
        return direct
    found, value = _json_field(text, "gs_value")
    if found:
        return _value(value)
    candidates = []
    for line in text.splitlines():
        candidate = _value(line)
        if candidate is not None:
            candidates.append(candidate)
        for match in VALUE_CUE.finditer(line):
            prefix = line[: match.start()]
            if re.search(r"\b(?:non|not)\b", prefix, re.IGNORECASE):
                continue
            tail = _unwrap(line[match.end() :])
            token = re.match(r"(" + NUMBER + r")(?=$|[\s.!;])", tail)
            if token:
                rest = tail[token.end() :]
                if re.search(NUMBER, rest):
                    continue
                candidate = _value(token.group(1))
                if candidate is not None:
                    candidates.append(candidate)
    unique = set(candidates)
    return next(iter(unique)) if len(unique) == 1 else None


def _category(text):
    if not isinstance(text, str):
        return None
    value = _unwrap(text).rstrip(".!;").strip().casefold()
    return value if value in CATEGORIES else None


def parse_gsi_category(response):
    text = _unwrap(_final_answer(response))
    direct = _category(text)
    if direct is not None:
        return direct
    found, value = _json_field(text, "gs_category")
    if found:
        return _category(value)
    candidates = []
    for line in text.splitlines():
        candidate = _category(line)
        if candidate:
            candidates.append(candidate)
        for match in CATEGORY_CUE.finditer(line):
            if re.search(r"\b(?:non|not)\b", line[: match.start()], re.IGNORECASE):
                continue
            tail = _unwrap(line[match.end() :])
            labels = re.findall(
                r"\b(?:" + "|".join(CATEGORIES) + r")\b", tail, re.IGNORECASE
            )
            if len(labels) == 1 and re.match(
                r"(?:" + "|".join(CATEGORIES) + r")\b", tail, re.IGNORECASE
            ):
                candidates.append(labels[0].casefold())
    unique = set(candidates)
    return next(iter(unique)) if len(unique) == 1 else None


def _gold(doc, field):
    mapping = doc.get(field)
    if not isinstance(mapping, dict) or mapping.get("Correct") not in mapping:
        raise ValueError("Missing or invalid GSI gold mapping: " + field)
    value = str(mapping[mapping["Correct"]])
    if field == "Values":
        if value not in ("0", "0.25", "0.5", "0.75", "1"):
            raise ValueError("Unexpected GSI gold value")
    elif value not in CATEGORIES:
        raise ValueError("Unexpected GSI gold category")
    return value


def _has_category(doc):
    if doc.get("Categories") is None:
        return False
    _gold(doc, "Categories")
    return True


def gsi_category_process_docs(dataset):
    # Only absent GOLD categories are excluded, never failed predictions.
    return dataset.filter(_has_category)


def gsi_task1_doc_to_target(doc):
    return _gold(doc, "Values")


def gsi_task2_doc_to_target(doc):
    return _gold(doc, "Categories")


def gsi_task1_process_results(doc, results):
    value = parse_gsi_value(_first_result(results))
    pair = (0.0 if value is None else value, float(_gold(doc, "Values")))
    return {
        **dict.fromkeys(
            ["gsi_mse", "gsi_mse_score", "gsi_nmse", "gsi_nmse_score", "gsi_ccc"], pair
        ),
    }


def gsi_task2_process_results(doc, results):
    value = parse_gsi_category(_first_result(results))
    pair = (value or "", _gold(doc, "Categories"))
    return {
        "gsi_f1_macro": pair,
        "gsi_f1_micro": pair,
    }


def _arrays(items):
    import numpy as np

    values = list(items)
    if not values:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    predictions, references = zip(*values)
    return np.asarray(references, dtype=float), np.asarray(predictions, dtype=float)


def aggregate_gsi_mse(items):
    import numpy as np

    truth, pred = _arrays(items)
    return float(np.mean((truth - pred) ** 2)) if truth.size else float("nan")


def aggregate_gsi_mse_score(items):
    return max(0.0, 1.0 - aggregate_gsi_mse(items))


def aggregate_gsi_nmse(items):
    import numpy as np

    truth, pred = _arrays(items)
    variance = np.var(truth)
    return (
        float(np.mean((truth - pred) ** 2) / variance) if variance > 0 else float("nan")
    )


def aggregate_gsi_nmse_score(items):
    nmse = aggregate_gsi_nmse(items)
    return float(1 / (1 + nmse)) if math.isfinite(nmse) and nmse >= 0 else float("nan")


def aggregate_gsi_ccc(items):
    import numpy as np

    truth, pred = _arrays(items)
    if not truth.size:
        return float("nan")
    mean_true, mean_pred = np.mean(truth), np.mean(pred)
    numerator = 2 * np.cov(truth, pred, ddof=0)[0, 1]
    denominator = (
        np.var(truth, ddof=0) + np.var(pred, ddof=0) + (mean_true - mean_pred) ** 2
    )
    return float(numerator / denominator) if denominator != 0 else 0.0


def _f1(items, average):
    from sklearn.metrics import f1_score

    pairs = list(items)
    if not pairs:
        return 0.0
    predictions, references = zip(*pairs)
    # No fixed labels= argument: exactly sklearn's union-of-observed-labels
    # behavior in the supplied scorer, including '' for invalid responses.
    return float(f1_score(references, predictions, average=average, zero_division=0))


def aggregate_gsi_f1_macro(items):
    return _f1(items, "macro")


def aggregate_gsi_f1_micro(items):
    return _f1(items, "micro")
