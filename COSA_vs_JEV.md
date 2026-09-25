# COSA vs. JEV

## JEV

- It takes a **state** (string, JSON or text array) plus typed questions, and returns typed answers with probabilities and a confidence value. It does not generate text.
- It has three question types:
  - **Choice** picks from declared options, with per-option probabilities.
  - **Score** rates against ordered levels, giving a distribution.
  - **Noul** answers yes/no with a probability.
- Several questions are evaluated in parallel in one call.
- Described as "transformer-based" and trained only on synthetic data, using RLCD (Reinforcement Learning for Calibrated Decisions). RLCD optimizes probabilities against outcomes, so that a stated 0.9 is right about 90% of the time.
- Claims: 70–500 ms responses, 40–200x faster and 40–400x cheaper than frontier LLMs. These figures are TypeSafe's own, from internal workflows. The weights and exact architecture are proprietary.

## COSA vs JEV

| | COSA | Jev |
|---|---|---|
| Core idea | Same: no text output, probabilities over supplied options | Same |
| Primitives | Choice only | Choice, Score, Noul |
| Input | Free text plus option strings | Structured state (JSON etc.) plus typed schema |
| Data | Synthetic, rule-generated | Synthetic, so the same strategy |
| Knowledge | Closed world: 17 drinks, 8 attributes, from scratch | General; likely built on a pretrained model (outside observers' guess) |
| Training objective | Cross-entropy on deterministic labels | RLCD: calibration is the explicit target |
| Calibration | Not addressed; outputs like 0.99999 | Central claim, but no published curves |
| Confidence value | None | Returned alongside probabilities |
| Parallel questions | One question per call | Many per call |
| Size and openness | 79k parameters, open, reproducible, 98.3% on held-out splits | Proprietary, undisclosed size |

## Assessment
- **COSA's choice mode is the same interface as Jev's.** The original goal from the first message, "output is a set of key-values where keys are the input options", is exactly Jev's Choice primitive. COSA is a small, open, from-scratch demonstration of that idea. It is not a Jev clone: what makes Jev valuable (general knowledge, calibration, scale) is what COSA PoC lacks.
- **Calibration is the biggest gap.** COSA labels are deterministic, so cross-entropy pushes probabilities toward 0 and 1. Jev's stated selling point is that probabilities mean something. A cheap step in that direction: add label noise or ambiguous cases, and measure expected calibration error, then temperature-scale on val. That would be a faithful mini-version of the idea. It would not be RLCD, which is undisclosed.
- **Missing primitives are easy to add.** Noul is a binary choice over `{true, false}`. Score is a choice over ordered levels, where COSA could add an ordinal loss. Both fit COSA architecture. Confidence is not a trivial addition, because how Jev defines it is not public.
- **Knowledge needs a pretrained encoder.** Jev handles arbitrary states because it is presumably built on a large pretrained model. COSA path B (frozen or fine-tuned sentence encoder) is the equivalent step.
- **Critics' point applies to both.** A constrained output format prevents invalid answers, not wrong ones.

Noul and Score primitives plus a calibration metric (ECE and temperature scaling) can be added to the PoC to make it closer to Jev's interface.

Sources:
- [Jev (AI model) - Wikipedia](https://en.wikipedia.org/wiki/Jev_(AI_model))
- [Jev Explained: Typesafe AI's Non-Autoregressive System-1 Model | MindStudio](https://www.mindstudio.ai/blog/jev-system-one-model-launch)
- [Jev: The Language Model That Won't Talk - Anthony Maio](https://anthonymaio.substack.com/p/jev-the-language-model-that-wont)
- [A new kind of AI model from a ChatGPT inventor is thrilling developers | TechCrunch](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/)
- [TypeSafe AI debuts model for machines that plays Doom | The Register](https://www.theregister.com/ai-and-ml/2026/09/16/typesafe-ai-debuts-model-for-machines-that-plays-doom/5296711)
