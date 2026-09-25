# Implementation Notes and Results

Proof-of-concept of the model in [system_formal_description.md](system_formal_description.md): given a text context and a set of candidate options, return a probability for each option (no text generation).

## 1. History of the design

1. **First spec (rejected).** The initial description was a causal single-layer attention followed by an MLP and a vocabulary-sized softmax. Review found it was a tiny GPT (next-token prediction over the vocabulary), not what was wanted, and not JEPA either (loss on tokens, no latent-space objective). It also had no positional encoding, residuals or feed-forward block.
2. **Clarified goal.** Input is a context plus options; output is `{option: probability}`. This is multiple-choice scoring: a dual-encoder / reranker. The spec was rewritten accordingly.
3. **Implementation** followed the rewritten spec (section 2).
4. **First training run** exposed a generalization problem (section 4), which led to a change of the data design.

## 2. What was built

| Component | File | Notes |
|---|---|---|
| Knowledge base | `optscore/data/kb.py` | 17 drinks, 8 attributes (alcoholic, fizzy, bitter, sweet, hot, sour, milky, cold), 1-4 surface words per attribute (e.g. fizzy = fizzy, carbonated, bubbly, sparkling) |
| Templates | `optscore/data/templates.py` | Statements (prefix x tail), questions, fillers |
| Tokenizer / vocab | `optscore/data/vocab.py` | Lowercase word tokenizer, apostrophes removed (`don't` -> `dont`), 81 tokens including `<pad>`, `<cls>`, `<unk>` |
| Generator | `optscore/data/generator.py` | Deterministic per seed; JSONL output |
| Batching | `optscore/data/batching.py` | Pads contexts and options; missing option slots become a lone `<cls>` and are masked |
| Model | `optscore/model/scorer.py` | As in the spec section 2 |
| Training / eval | `optscore/training/` | AdamW, warmup + cosine schedule, gradient clipping |
| Inference | `optscore/inference.py`, `optscore/checkpoint.py` | `Scorer.probabilities(context, options)` |
| CLI | `main.py` | `gen`, `train`, `eval`, `ask` |

### Model
- Shared bidirectional Transformer encoder (no causal mask), post-LN, GELU, padding mask.
- d_model 64, 4 heads, 2 layers, FFN 128, max length 48, dropout 0.1. **79,360 parameters.**
- `<cls>` hidden state -> linear projection -> L2 normalisation. Context and each option go through the same encoder.
- Score = cosine / tau with tau = 0.1 (fixed). Softmax runs over the options of that sample only, so the option count may vary (2 to N at inference; trained on 2 and 3).
- Loss: cross-entropy over the options. In-batch negatives were **not** used: with 17 drinks the same drink is correct in many samples, which would create false negatives.

### Synthetic data
Each sample is a context (statement + question, optional filler, random order), options, and the index of the correct one.

- **Statement:** `{prefix} {attribute word} {tail}`, e.g. `i dont like carbonated beverages` (negative) or `i love sweet drinks` (positive). 7 negative and 7 positive prefixes, 4 tails.
- **Question:** `should i take a beer or a wine`, `which one ...`, etc. Options are rendered with or without articles.
- **Label rule:** negative polarity -> the one option **without** the attribute; positive -> the one option **with** it. Only option sets with exactly one valid answer are generated, so labels are never ambiguous.
- **Mix:** 20% three-option samples, 50/50 polarity, options shuffled (label position is balanced), 40% get a neutral filler sentence.
- **Contrast:** the same option set and attribute occur with both polarities and opposite answers, so "mention of an attribute = choose it" is not a valid shortcut.

### Splits (default sizes 20000 / 2000 / 2000 / 2000)

| Split | Option sets | Phrasing |
|---|---|---|
| `train` | seen | training templates |
| `val` | held-out drink pairs | training templates |
| `test` | held-out drink pairs (disjoint from val) | training templates |
| `test_templates` | test option sets | **held-out** statement combinations and question forms |

A drink-option set belongs to val/test if any of its pairs was held out (about 10% / 15% of the 136 pairs). Every drink still appears in training, only in other combinations. Held-out templates contain only words seen in training (checked by a test).

## 3. Testing

`pytest` runs 41 fast tests (about 10 s); `pytest -m slow` adds 3 tests that train the full PoC (about 2m40 on 8 CPU cores).

- **Knowledge base:** attributes valid, synonyms unique, each attribute discriminates.
- **Generator:** label correctness recomputed independently from the KB, exactly one valid answer, option sets disjoint across splits, train never uses held-out templates, every prefix/tail is seen in training, label position balance, contrastive polarity pairs exist, determinism per seed, JSONL round trip.
- **Vocab:** apostrophe handling, unknown-token mapping, truncation, no unknown tokens in any generated sample, held-out templates use only seen words.
- **Model:** shapes, unit-norm encodings, masked slots get probability 0 with no NaNs, result unchanged by padding or by batch composition, permuting the options permutes the probabilities, gradients reach every parameter, encoder is bidirectional.
- **Training / inference:** loss decreases, reproducible under a fixed seed, checkpoint round trip gives identical outputs, JSON-serialisable probabilities summing to 1, any option count accepted, invalid input rejected.
- **Slow (full PoC):** test accuracy > 0.95, held-out-template accuracy > 0.85, and the spec example answered correctly in both polarities.

## 4. Results

### First run: hand-written templates
13 training statement templates and 6 held-out ones.

| Split | Accuracy |
|---|---|
| val / test (unseen drink sets) | 99.95% / 98.95% |
| test_templates (unseen phrasing) | **66.4%** |

Breakdown by held-out statement template: four scored 97-99%, two scored about 0% (`i cannot stand {a} drinks` 4.7%, `i want {a} beverages please` 0.3%). The failures were systematic reversals, not noise. Question forms were not the problem. Cause: with only 13 phrasings the model learned associations tied to specific word combinations (e.g. the tail word `please` or `drinks` appears under both polarities in training) rather than the polarity of the verb phrase.

This is a **data-diversity** issue, not a code bug: the model generalised perfectly over unseen drink combinations but not over unseen phrasings.

### Fix: compositional statements
Statements became `prefix x tail` (7 prefixes x 4 tails per polarity = 28 combinations). A fixed subset (`(prefix_index + tail_index) % 4 == 3`, 7 per polarity) is held out; 21 per polarity remain for training. Every prefix and tail is still seen in training, so the model must learn each prefix's polarity independently of the tail.

### Final run (seed 0, 20000 training samples, 12 epochs)

| Split | Accuracy | NLL | Pos-polarity | Neg-polarity | 2 options | 3 options |
|---|---|---|---|---|---|---|
| val | 94.9% | 0.172 | 93.5% | 96.3% | 94.5% | 96.3% |
| test | 98.3% | 0.038 | 97.9% | 98.7% | 98.5% | 97.7% |
| test_templates | 98.3% | 0.041 | 98.5% | 98.2% | 98.7% | 96.8% |

Chance is 50% (2 options) or 33% (3 options).

Training loss falls from 0.77 to 0.03 over 12 epochs; validation accuracy jumps between epochs 3 and 4 (66% -> 92%) once the model picks up the attribute/drink relation, then plateaus near 95%.

### Example queries

| Context | Options | Output |
|---|---|---|
| Should I take a beer or a wine? I don't like carbonated beverages | beer, wine | beer 0.004, wine 0.996 |
| Should I take a beer or a wine? I want something carbonated | beer, wine | beer 0.995, wine 0.005 |
| I hate sweet stuff. which one cola, coffee or juice | cola, coffee, juice | cola 3e-6, coffee 0.99999, juice 7e-7 |

## 5. Observations and limitations

- **Val is lower than test (94.9% vs 98.3%).** Both use held-out drink pairs; the specific pairs in val are harder (probably drinks whose attributes only differ in one attribute that rarely appears). Not investigated further.
- **Overconfident probabilities.** Outputs like 0.99999 reflect the clean synthetic labels, not calibrated uncertainty. Temperature scaling on a validation set is the standard remedy.
- **Closed world.** The model knows only the 17 drinks and 8 attributes in the knowledge base, and the attribute words in the templates. Unknown words map to `<unk>` and carry no information. Answering e.g. "champagne or grape juice" style questions about unseen items requires pretrained embeddings.
- **Phrasing generalisation is limited to recombination.** It handles new combinations of known prefixes, tails and question forms; entirely new wording (new verbs, new sentence structures) is untested and would need more diverse data.
- **Only single-attribute constraints.** Contexts state one attribute; multiple constraints, conflicting constraints or questions with no valid answer are not modelled.
- **Speed.** About 2m40 for the full run on CPU, roughly 85 ms per step; no GPU needed.

## 6. Suggested next steps

1. Calibrate probabilities (temperature scaling on val).
2. Add multi-constraint contexts ("no fizzy and no alcohol") and options with no valid answer.
3. Replace the from-scratch embeddings with a pretrained embedding/encoder to handle unseen words and real-world knowledge; re-check negation handling, which embedding similarity typically gets wrong.
4. Try the cross-encoder variant (context and option in one sequence) and compare accuracy on negation.
5. Investigate the val vs. test gap.
