# COSA: Contextual Option Scoring via Attention

Proof-of-concept: an attention model that reads a text and a list of options and returns a probability for each option, instead of generating text.

```
$ python3 main.py ask "Should I take a beer or a wine? I don't like carbonated beverages" beer wine
{
  "beer": 0.0036,
  "wine": 0.9964
}
```

The model only knows a small synthetic world (17 drinks, 8 attributes such as fizzy, sweet, hot).

## Setup

Requires Python 3 and about 1 GB of disk for `.venv` (CPU build of PyTorch). No GPU needed.

```
./setup.sh
```

This creates `.venv` and installs `requirements.txt` (torch CPU, numpy, pytest).

## Usage

All commands are run from the project directory.

### Train (generates the dataset if missing)
```
.venv/bin/python3 main.py train
```
Takes about 3 minutes on 8 CPU cores. Writes `datasets/*.jsonl` and `checkpoints/model.pt`, then prints accuracy on `val`, `test` and `test_templates`.

Options: `--epochs 12`, `--n-train 20000`, `--seed 0`, `--data datasets`, `--ckpt checkpoints/model.pt`.

### Generate only the dataset
```
.venv/bin/python3 main.py gen [--n-train 20000] [--seed 0] [--data datasets]
```
Creates `train`, `val`, `test`, `test_templates` JSONL files, one sample per line:
```json
{"context": "which one cocoa, latte or seltzer no warm drinks", "options": ["cocoa", "latte", "seltzer"], "label": 2, "attr": "hot", "polarity": "neg", "statement_tpl": "no {a} drinks", "question_tpl": "which one {list}"}
```

### Evaluate a saved model
```
.venv/bin/python3 main.py eval
```

### Ask
```
.venv/bin/python3 main.py ask "<context>" <option> <option> [<option> ...]
```
Prints JSON `{option: probability}`. Two or more distinct options; the probabilities sum to 1. Use the option names as they appear in the context, one word each (e.g. `beer`).

### From Python
```python
from optscore.inference import Scorer

s = Scorer.from_checkpoint("checkpoints/model.pt")
s.probabilities("I hate sweet stuff. which one cola, coffee or juice", ["cola", "coffee", "juice"])
# {'cola': 3e-06, 'coffee': 0.99999, 'juice': 7e-07}
```

## What it understands

- **Drinks (options):** beer, wine, champagne, whiskey, cider, cola, seltzer, kombucha, lemonade, water, juice, coffee, tea, latte, cocoa, milk, milkshake.
- **Attributes (words in the context):** alcoholic/boozy, fizzy/carbonated/bubbly/sparkling, bitter, sweet/sugary, hot/warm, sour/tangy, milky/creamy, cold/chilled/icy.
- **Polarity:** positive (`i want`, `i like`, `i love`, `i prefer`, `i crave`, `give me`, `i need`) picks the option that has the attribute; negative (`i dont like`, `i hate`, `i cannot stand`, `i dislike`, `no`, `please avoid`, `nothing`) picks the option that lacks it.
- One attribute per context. Words outside the vocabulary are ignored, and drinks outside the list carry no meaning to the model.

Full details in [optscore/data/kb.py](optscore/data/kb.py) and [optscore/data/templates.py](optscore/data/templates.py). Extend the world by editing those files and retraining.

## Tests

```
.venv/bin/python3 -m pytest              # 41 fast tests, about 10 s
.venv/bin/python3 -m pytest -m slow      # trains the full PoC and checks accuracy, about 3 min
```

## Layout

```
main.py                    CLI (gen, train, eval, ask)
setup.sh                   creates .venv, installs requirements
optscore/
  data/                    kb.py, templates.py, vocab.py, generator.py, batching.py
  model/scorer.py          bidirectional attention encoder + option scoring
  training/                train.py, evaluate.py
  checkpoint.py            save / load model + vocab
  inference.py             Scorer.probabilities(context, options)
tests/                     unit, generator, model and training tests
datasets/  checkpoints/    generated output (gitignored)
```
