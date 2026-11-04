# GSI:detect

GSI:detect evaluates the detection and classification of gender stereotypes in
Italian texts. This implementation adapts its two tasks to generative evaluation
in Evalita-LLM.

## Tasks and prompts

**Task 1** asks for a stereotype value between 0 and 1. The gold averages four
binary annotations, giving five possible values: 0, 0.25, 0.5, 0.75 and 1.
The model can predict any number in the interval, without rounding to a gold level.

**Task 2** asks for one category: `role`, `personality`, `competence`, `physical`,
`sexual` or `relational`. Items without a gold category are excluded from this
task only.

Each task has two prompts: P1 without an introductory task description and P2
with it. Both request a concise answer. The parsers also accept an unambiguous
answer surrounded by text or common formatting.

## Data and examples

The [adapted dataset](https://huggingface.co/datasets/evalitahf/gsi_detect)
stores each text, value and category in one JSONL record. Multiple-choice mappings
store the gold labels; evaluation uses generated numbers and categories.

TEST contains 810 texts for Task 1 and 568 categorized texts for Task 2.
Missing categories are represented as `null`.

The `dev_fewshot` split contains six DEV examples. The first five cover all gold
values. Filtering out the example without a category leaves five valid category
examples. Each subtask therefore receives five examples in five-shot evaluation.

## Metrics

Task 1 follows the original scoring script:

- **MSE**: mean squared error between predictions and gold values.
- **MSE score**: `max(0, 1 - MSE)`.
- **NMSE**: MSE divided by the variance of the gold values.
- **NMSE score**: `1 / (1 + NMSE)`, the original ranking metric.
- **CCC**: agreement accounting for correlation, mean and scale.

Task 2 reports **macro-F1** and **micro-F1**. The original ranking uses micro-F1.

Invalid numerical responses become 0, and invalid categories become an empty
label, following the original script. These predictions remain in the metrics.
A fallback of 0 can therefore match a gold value of 0. Category comparisons
ignore case. Parsing and scoring are defined in `gsi_utils.py`.

## Usage

With a Hugging Face backend, install the harness with `pip install -e .`.

```bash
lm_eval --model hf --model_args pretrained=MODEL --tasks evalita-mp_gsi_detect --num_fewshot 0 --apply_chat_template --batch_size 1
```

For five-shot evaluation, use `--num_fewshot 5` and add `--fewshot_as_multiturn`.
The default is zero-shot.

## Reference

Comandini et al. (2026). [GSI:detect at EVALITA 2026: Overview of the Task on
Detecting Gender Stereotypes in Italian](https://aclanthology.org/2026.evalita-1.11/).
[Official task website](https://gsi-d-evalita.fbk.eu/).

Dataset adaptation for Evalita-LLM: Pietro Botarelli. The adapted dataset is
released under CC-BY-NC-SA-4.0.
