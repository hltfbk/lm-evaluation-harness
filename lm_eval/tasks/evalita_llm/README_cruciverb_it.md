# Cruciverb-IT

Cruciverb-IT evaluates Italian crossword solving. This implementation adapts
its two tasks to generative evaluation in Evalita-LLM.

## Tasks and prompts

**Task 1** asks the model to solve a clue, given the answer length. It accepts
one answer or a list of candidates, ordered from most to least likely.

**Task 2** asks for a completed crossword. The prompt shows a text grid: `_`
marks an empty white cell, `#` a black cell, and numbers identify starting cells.
Clues are grouped under ORIZZONTALI and VERTICALI. If the model cannot format
the grid, it can return answers with their numbers and directions.

Each task has two prompts: P1 without an introductory task description and P2
with it. Test gold answers are used only for scoring.

## Data and examples

The [adapted dataset](https://huggingface.co/datasets/evalitahf/cruciverb_it)
contains JSONL records. Task 1 has nested test subsets of 1,000, 2,000 and 5,000
clues; the default is `test_5k`. Five `dev_fewshot` examples show single answers
and candidate lists. The original gold word remains the scoring reference.

Task 2 has 500 training, 50 development and 50 test crosswords. Five-shot examples
come from `train`: numbered solutions under ORIZZONTALI and VERTICALI are
followed by the completed text grid, showing both accepted formats.

## Metrics

Task 1 reports **Acc@1**, **Acc@10** and **MRR@10** (`mrr`). MRR@10 averages the
reciprocal rank of the gold within the first ten candidates. It corrects an
indexing mistake in the original script, which truncated the gold word instead
of the candidate list.

Task 2 retains the original metrics:

- **Word accuracy**: correct words read from the grid, divided by all clues.
- **Character accuracy**: correct letters divided by the gold white cells.
- **Full Match**: the proportion of completely correct grids.

**Direct word accuracy** compares answers with their gold words before grid
reconstruction. A correct answer keeps its credit if another answer conflicts
with it. When both formats are returned, the original metrics use the grid and
direct word accuracy uses the list. With only a grid, the two word accuracies agree.

Task 2 scores are averaged across grids. Wrong-length answers are not inserted;
conflicting cells remain empty. Comparisons ignore case and use Unidecode.
Invalid outputs remain in evaluation and receive zero. The implementation is
in `cruciverb_utils.py`.

## Usage

Install the dependency with `pip install -e ".[evalita]"`.

```bash
lm_eval --model hf --model_args pretrained=MODEL --tasks evalita-mp_cruciverb_it --num_fewshot 0 --apply_chat_template --batch_size 1
```

For five-shot evaluation, use `--num_fewshot 5` and add `--fewshot_as_multiturn`.
The default is zero-shot.

## Reference

Ciaccio et al. (2026). [Cruciverb-IT at EVALITA 2026: Overview of the Crossword
Solving in Italian Task](https://aclanthology.org/2026.evalita-1.36/).

Dataset adaptation for Evalita-LLM: Pietro Botarelli. The data retain the
research-use conditions of the [original dataset](https://huggingface.co/datasets/cruciverb-it/evalita2026).
