"""Offline regression tests for the two generative Evalita-LLM task families."""

import copy
import importlib.util
import math
from pathlib import Path

import pytest


TASKS = Path(__file__).resolve().parents[2] / "lm_eval/tasks/evalita_llm"


def load(name, path):
    # Like the 0.4.8 YAML loader, do not rely on sys.path or sys.modules changes.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cruci():
    pytest.importorskip(
        "unidecode", reason="Install lm_eval[evalita] to run Cruciverb scoring tests."
    )
    return load("tested_cruciverb_utils", TASKS / "cruciverb_utils.py")


@pytest.fixture(scope="module")
def gsi():
    return load("tested_gsi_utils", TASKS / "gsi_utils.py")


@pytest.fixture
def square():
    return {
        "id": "synthetic",
        "empty_grid": [[" ", " "], [" ", " "]],
        "gold_grid": [["A", "B"], ["C", "D"]],
        "clues": [
            {
                "clue": "First row",
                "row": 0,
                "col": 0,
                "direction": "A",
                "length": 2,
                "target": "AB",
            },
            {
                "clue": "Second row",
                "row": 1,
                "col": 0,
                "direction": "A",
                "length": 2,
                "target": "CD",
            },
            {
                "clue": "First column",
                "row": 0,
                "col": 0,
                "direction": "D",
                "length": 2,
                "target": "AC",
            },
            {
                "clue": "Second column",
                "row": 0,
                "col": 1,
                "direction": "D",
                "length": 2,
                "target": "BD",
            },
        ],
    }


def gsi_doc(value="0.5", category="role"):
    values = dict(zip("ABCDE", ["0", "0.25", "0.5", "0.75", "1"]))
    values["Correct"] = next(k for k, v in values.items() if v == value)
    categories = dict(
        zip(
            "ABCDEF",
            ["role", "personality", "competence", "physical", "sexual", "relational"],
        )
    )
    categories["Correct"] = next(k for k, v in categories.items() if v == category)
    return {
        "Id": 1,
        "Question": "Synthetic text.",
        "Values": values,
        "Categories": categories,
    }


@pytest.mark.parametrize(
    "response,expected",
    [
        ("LIRA", ["LIRA"]),
        ("euro, lira, marco", ["euro", "lira", "marco"]),
        ("Risposta: LIRA — moneta cipriota", ["LIRA"]),
        ('["EURO", "LIRA"]', ["EURO", "LIRA"]),
        ("1. EURO\n2. LIRA", ["EURO", "LIRA"]),
        ("Non è LIRA", []),
        ("La definizione cita LIRA ma non propongo risposte.", []),
        ("<think>LIRA</think>EURO", ["EURO"]),
        ("<think>LIRA", []),
        ("ragiono\n</think> LIRA", ["LIRA"]),
        ("<|channel>thought\nLIRA<channel|>EURO<turn|>", ["EURO"]),
        ("<|channel>thought\nLIRA", []),
        ({"raw_output": "LIRA", "reasoning": "LIRA"}, []),
        ({"final_answer": "lira", "reasoning": "euro"}, ["lira"]),
    ],
)
def test_task1_parser(cruci, response, expected):
    assert cruci.parse_task1_candidates(response) == expected


@pytest.mark.parametrize(
    "candidates,metrics",
    [
        ("LIRA", (1, 1, 1)),
        ("EURO, lira", (0, 1, 0.5)),
        (", ".join(["EURO"] * 9 + ["LIRA"]), (0, 1, 0.1)),
        (", ".join(["EURO"] * 10 + ["LIRA"]), (0, 0, 0)),
        ("EURO, EURO, LIRA", (0, 1, 1 / 3)),
        ("EUROOOOOO, LIRA", (0, 1, 0.5)),
        ("", (0, 0, 0)),
    ],
)
def test_task1_rank_without_filtering(cruci, candidates, metrics):
    result = cruci.cruciverb_task1_process_results(
        {"answer": "lira", "answer_length": 4}, [candidates]
    )
    assert tuple(result.values()) == metrics


def test_task1_accent_and_long_answer(cruci):
    assert (
        cruci.cruciverb_task1_process_results({"answer": "città"}, ["CITTA"])["acc@1"]
        == 1
    )
    assert (
        cruci.cruciverb_task1_process_results(
            {"answer": "PRECIPITEVOLISSIMEVOLMENTE"}, ["precipitevolissimevolmente"]
        )["mrr"]
        == 1
    )


@pytest.mark.parametrize("target", [None, "", "EURO, LIRA"])
def test_task1_nullable_demonstration_target(cruci, target):
    doc = {"answer": "LIRA", "fewshot_target": target}
    assert cruci.cruciverb_task1_doc_to_target(doc) == (target or "LIRA")
    assert cruci.cruciverb_task1_doc_to_target({"answer": "LIRA"}) == "LIRA"


@pytest.mark.parametrize(
    "response",
    [
        "A B\nC D",
        "[['A','B'],['C','D']]",
        "```\nA B\nC D\n```",
        "ORIZZONTALI\n1. AB\n3. CD\nVERTICALI\n1. AC\n2. BD",
    ],
)
def test_task2_perfect_formats(cruci, square, response):
    metrics = cruci.cruciverb_task2_process_results(square, [response])
    assert all(value == 1 for value in metrics.values())


@pytest.mark.parametrize("response", ["", "A B C\nD E F", "<think>A B\nC D", "None"])
def test_task2_failed_outputs_retained(cruci, square, response):
    assert all(
        value == 0
        for value in cruci.cruciverb_task2_process_results(square, [response]).values()
    )


def test_task2_no_oracle_and_gold_independence(cruci, square):
    response = "ORIZZONTALI\n1. AB\n3. ZZ\nVERTICALI\n1. AC\n2. ZZ"
    first = cruci.parse_task2_output(response, square).to_dict()
    changed = copy.deepcopy(square)
    changed["gold_grid"] = [["Z", "Z"], ["Z", "Z"]]
    for clue in changed["clues"]:
        clue["target"] = "ZZ"
    assert cruci.parse_task2_output(response, changed).to_dict() == first
    assert (
        cruci.cruciverb_task2_process_results(square, [response])["full_match_accuracy"]
        == 0
    )
    assert first["answers"] and any(v == "ZZ" for v in first["answers"].values())


def test_task2_generation_never_reads_gold(cruci, square):
    first = cruci.cruciverb_task2_prompt1(square)
    changed = copy.deepcopy(square)
    changed.pop("gold_grid")
    for clue in changed["clues"]:
        clue.pop("target")
    assert cruci.cruciverb_task2_prompt1(changed) == first
    assert "1 2\n3 _" in first
    assert "target" not in first and "gold_grid" not in first


@pytest.mark.parametrize(
    "response,expected",
    [
        ("0", 0),
        ("1", 1),
        ("0.37", 0.37),
        ("0,75", 0.75),
        (".25", 0.25),
        ("7.5e-1", 0.75),
        ("**0.5**", 0.5),
        ('"0.5"', 0.5),
        ("0.75.", 0.75),
        ("Risposta: 0.6", 0.6),
        ('{"gs_value": 0.42}', 0.42),
        ("Testo di contorno\n0.3", 0.3),
        ("0.5\n0.7", None),
        ("Risposta: 0.5 oppure 0.7", None),
        ("1.2", None),
        ("-0.2", None),
        ("NaN", None),
        ("Infinity", None),
        ("75%", None),
        ("A", None),
        ("", None),
        ("<think>0.9</think>0.2", 0.2),
        ("<think>0.9", None),
        ('{"gs_value":true}', None),
        ("Non propongo un valore.", None),
    ],
)
def test_gsi_continuous_parser(gsi, response, expected):
    assert gsi.parse_gsi_value(response) == expected


@pytest.mark.parametrize(
    "response,expected",
    [
        ("role", "role"),
        ("COMPETENCE", "competence"),
        ("**sexual**", "sexual"),
        ("relational.", "relational"),
        ("Categoria: physical", "physical"),
        ("La categoria è personality", "personality"),
        ('{"gs_category":"role"}', "role"),
        ("role oppure sexual", None),
        ("role\nsexual", None),
        ("Non è role", None),
        ("Il testo cita la parola role.", None),
        ("A", None),
        ("no", None),
        ("", None),
        ("<think>role</think>competence", "competence"),
        ("<|channel>thought\nrole<channel|>physical", "physical"),
    ],
)
def test_gsi_category_parser(gsi, response, expected):
    assert gsi.parse_gsi_category(response) == expected


def test_gsi_invalids_use_official_replacements(gsi):
    doc = gsi_doc("0.75", "physical")
    regression = gsi.gsi_task1_process_results(doc, ["bad response"])
    assert regression["gsi_mse"] == (0, 0.75)
    assert regression["gsi_ccc"] == (0, 0.75)
    assert set(regression) == {
        "gsi_mse",
        "gsi_mse_score",
        "gsi_nmse",
        "gsi_nmse_score",
        "gsi_ccc",
    }
    category = gsi.gsi_task2_process_results(doc, ["bad response"])
    assert category["gsi_f1_macro"] == ("", "physical")
    assert set(category) == {"gsi_f1_macro", "gsi_f1_micro"}


def test_gsi_macro_uses_observed_labels_like_organizer(gsi):
    assert gsi.aggregate_gsi_f1_macro([("role", "role")]) == 1
    assert math.isclose(
        gsi.aggregate_gsi_f1_macro([("", "role"), ("role", "role")]), 1 / 3
    )
    assert gsi.aggregate_gsi_f1_micro([("", "role"), ("role", "role")]) == 0.5


def test_gsi_category_gold_filter(gsi):
    from datasets import Dataset

    docs = [gsi_doc(), {**gsi_doc(), "Categories": None}]
    result = gsi.gsi_category_process_docs(Dataset.from_list(docs))
    assert len(result) == 1


def test_prompt_numbering_and_width_above_nine(cruci):
    n = 7
    doc = {"empty_grid": [[" "] * n for _ in range(n)], "clues": []}
    for row in range(n):
        doc["clues"].append(
            {"clue": "Across", "row": row, "col": 0, "direction": "A", "length": n}
        )
    for col in range(n):
        doc["clues"].append(
            {"clue": "Down", "row": 0, "col": col, "direction": "D", "length": n}
        )
    text = cruci.cruciverb_task2_prompt1(doc)
    grid = text.split("\n\nDEFINIZIONI")[0].rsplit("\n\n", 1)[1].splitlines()
    assert len(grid) == n and len(set(map(len, grid))) == 1
    assert "10" in text and "13" in text


def test_gsi_regression_known_values(gsi):
    pairs = [(0.0, 0.0), (0.5, 0.5), (0.5, 1.0)]
    assert gsi.aggregate_gsi_mse(pairs) == pytest.approx(1 / 12)
    assert gsi.aggregate_gsi_mse_score(pairs) == pytest.approx(11 / 12)
    assert gsi.aggregate_gsi_nmse(pairs) == pytest.approx(0.5)
    assert gsi.aggregate_gsi_nmse_score(pairs) == pytest.approx(2 / 3)
    assert gsi.aggregate_gsi_ccc(pairs) == pytest.approx(2 / 3)


def test_gsi_constant_gold_and_invalid_zero(gsi):
    pairs = [(0.5, 0.5), (0.5, 0.5)]
    assert math.isnan(gsi.aggregate_gsi_nmse(pairs))
    assert math.isnan(gsi.aggregate_gsi_nmse_score(pairs))
    assert gsi.aggregate_gsi_ccc(pairs) == 0
    scores = gsi.gsi_task1_process_results(gsi_doc("0"), ["invalid"])
    assert scores["gsi_mse"] == (0.0, 0.0)
    assert gsi.aggregate_gsi_mse([scores["gsi_mse"]]) == 0


def test_task2_grid_metrics_and_partial_credit(cruci, square):
    scores = cruci.cruciverb_task2_process_results(square, ["A B\nC Z"])
    assert scores == {
        "word_acc_direct": 0.5,
        "word_acc": 0.5,
        "char_acc": 0.75,
        "full_match_accuracy": 0,
    }
    assert (
        cruci.cruciverb_task2_process_results(square, ["a b\nc d"])[
            "full_match_accuracy"
        ]
        == 1
    )
    # The missing down answer can be reconstructed by crossings in a grid.
    output = "ORIZZONTALI\n1. AB\n3. CD\nVERTICALI\n1. AC"
    scores = cruci.cruciverb_task2_process_results(square, [output])
    assert scores["word_acc_direct"] == 0.75
    assert (
        scores["word_acc"] == scores["char_acc"] == scores["full_match_accuracy"] == 1
    )


def test_gold_mappings_do_not_change_gsi_predictions(gsi):
    first = gsi_doc("0", "role")
    second = gsi_doc("1", "sexual")
    assert gsi.gsi_task1_process_results(first, ["0.37"])["gsi_mse"][0] == 0.37
    assert gsi.gsi_task1_process_results(second, ["0.37"])["gsi_mse"][0] == 0.37
    assert (
        gsi.gsi_task2_process_results(first, ["physical"])["gsi_f1_macro"][0]
        == "physical"
    )
    assert (
        gsi.gsi_task2_process_results(second, ["physical"])["gsi_f1_macro"][0]
        == "physical"
    )


def test_reasoning_is_not_a_missing_final_fallback(cruci, gsi):
    raw = {"raw_output": "LIRA", "reasoning": "LIRA"}
    assert cruci.parse_task1_candidates(raw) == []
    assert gsi.parse_gsi_value({"raw_output": "0.5", "reasoning": "0.5"}) is None
    assert gsi.parse_gsi_category({"raw_output": "role", "reasoning": "role"}) is None
    assert (
        gsi.parse_gsi_category(
            "<|channel|>analysis<|message|>role<|end|><|channel|>final<|message|>sexual<|return|>"
        )
        == "sexual"
    )
