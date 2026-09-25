# Formal Description: Attention-Based Option Scoring Model

## 1. Overview
This document specifies a model that receives a text **context** and a set of candidate **options** (also text), and returns a probability distribution over those options. There is no text generation. The output is a mapping `option -> probability`.

Example:

```
context: "Should I take a beer or a wine? I don't like carbonated beverages."
options: ["beer", "wine"]
output : {"beer": 0.1, "wine": 0.9}
```

Architecture type: **dual-use bidirectional attention encoder** (a shared encoder embeds context and options) with **dot-product scoring** and a **softmax over the options** only.

---

## 2. Mathematical Formalism

### 2.1 Vocabulary and Inputs
Let $V$ be a finite vocabulary of size $|V|$, including special tokens `<PAD>` and `<CLS>`.

A text of length $T$ is a sequence of token indices $\mathbf{x} = [x_1, \dots, x_T]$, $x_t \in \{1,\dots,|V|\}$.

A sample consists of:
- a context $\mathbf{x}^{c}$,
- $N$ options $\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(N)}$ ($N$ may vary between samples, $N \ge 2$),
- a label $y \in \{1,\dots,N\}$ (training only): the index of the correct option.

### 2.2 Embedding
Token embedding $E \in \mathbb{R}^{|V| \times d}$ plus positional embedding $P \in \mathbb{R}^{T_{max} \times d}$. A `<CLS>` token is prepended to every text, so $x_0 = \texttt{<CLS>}$:

$$\mathbf{u}_t = E[x_t] + P[t], \qquad \mathbf{U} = [\mathbf{u}_0, \dots, \mathbf{u}_T] \in \mathbb{R}^{(T+1) \times d}$$

Embeddings may be trained from scratch or initialized from a pretrained embedding model.

### 2.3 Encoder (bidirectional self-attention)
$L$ stacked layers, each with multi-head self-attention and a feed-forward network, both with residual connections and layer normalization:

$$\mathbf{Q} = \mathbf{U}\mathbf{W}_Q,\quad \mathbf{K} = \mathbf{U}\mathbf{W}_K,\quad \mathbf{V} = \mathbf{U}\mathbf{W}_V$$

$$\text{Attn}(\mathbf{U}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}_{pad}\right)\mathbf{V}$$

$$\mathbf{U}' = \text{LN}(\mathbf{U} + \text{Attn}(\mathbf{U})), \qquad \mathbf{U}'' = \text{LN}(\mathbf{U}' + \text{FFN}(\mathbf{U}'))$$

$\mathbf{M}_{pad}$ masks `<PAD>` positions with $-\infty$. **No causal mask is used**: every token sees every other token, because the task is understanding, not autoregressive generation.

The encoder $\text{Enc}(\cdot)$ returns the final hidden state of the `<CLS>` position, projected and normalized:

$$\mathbf{r} = \frac{\mathbf{W}_p\, \mathbf{h}_{CLS}}{\lVert \mathbf{W}_p\, \mathbf{h}_{CLS} \rVert} \in \mathbb{R}^{d}$$

### 2.4 Context and Option Representations
The same encoder (shared weights) is applied to each text:

$$\mathbf{c} = \text{Enc}(\mathbf{x}^{c}), \qquad \mathbf{o}_i = \text{Enc}(\mathbf{x}^{(i)}),\quad i = 1..N$$

Variant: separate encoders for context and options (bi-encoder with untied weights).

### 2.5 Scoring and Probabilities
$$s_i = \frac{\mathbf{c}^\top \mathbf{o}_i}{\tau}$$

$\tau > 0$ is a temperature (fixed, e.g. 0.1, or learned). Since $\mathbf{c}$ and $\mathbf{o}_i$ are unit vectors, $s_i$ is a scaled cosine similarity.

$$P(i \mid \mathbf{x}^{c}, \{\mathbf{x}^{(j)}\}) = \frac{\exp(s_i)}{\sum_{j=1}^{N} \exp(s_j)}$$

The softmax runs over the **options of that sample**, not over the vocabulary. The number of options is arbitrary at inference time.

### 2.6 Objective
Cross-entropy over the options:

$$\mathcal{L}(\Theta) = -\log P(y \mid \mathbf{x}^{c}, \{\mathbf{x}^{(j)}\})$$

**In-batch negatives (optional, not used in the PoC):** for a batch of $B$ samples, the correct options of the other samples are appended as extra negative candidates for each sample. This increases the number of negatives at no data cost, but with a tiny option vocabulary it creates false negatives (the same drink is correct in many samples), so the PoC omits it.

### 2.7 Inference
1. Tokenize context and options.
2. Compute $\mathbf{c}$ and each $\mathbf{o}_i$.
3. Compute $P(i)$ as in 2.5.
4. Return `{option_text_i: P(i)}` (JSON).

Since $\mathbf{o}_i$ does not depend on the context, option vectors can be precomputed and cached.

---

## 3. Concrete Example

Vocabulary (subset): `<PAD> <CLS> should i take a beer or wine ? dont like carbonated beverages`

Sample:
- Context: `["should","i","take","a","beer","or","a","wine","?","i","dont","like","carbonated","beverages"]`
- Options: `[["beer"], ["wine"]]`
- Label: `2` (wine)

Data flow:

```
context tokens ─► [CLS]+embed+pos ─► bidirectional attention x L ─► h_CLS ─► proj+norm ─► c
"beer"         ─► [CLS]+embed+pos ─► bidirectional attention x L ─► h_CLS ─► proj+norm ─► o_1
"wine"         ─► [CLS]+embed+pos ─► bidirectional attention x L ─► h_CLS ─► proj+norm ─► o_2
                                          s_i = c·o_i / τ
                                          softmax over {1,2}
                                   ─►  {"beer": 0.1, "wine": 0.9}
```

The model can only answer this correctly if training data teaches (a) that beer is carbonated and wine is not, and (b) that "dont like X" excludes options having attribute X.

---

## 4. Design Notes and Limitations

1. **Knowledge comes from training data.** A model trained from scratch knows only the facts and patterns in its dataset. Real-world knowledge requires a pretrained encoder (frozen or fine-tuned).
2. **Negation is hard.** "I don't like X" and "I like X" are lexically close. Training data must contain both polarities with balanced labels, or the model will learn the shortcut "mentioned attribute = good".
3. **Generalization must be tested on held-out combinations** (option pairs or attribute-option pairs never seen together), otherwise memorization is indistinguishable from learning.
4. **Calibration.** Softmax probabilities reflect the training distribution and can be overconfident. Temperature scaling on a validation set can be applied post hoc.
5. **Not a generative model.** It cannot produce options, only rank the ones supplied.
6. **Relation to other approaches.** This is a dual-encoder (bi-encoder) reranker with a contrastive/cross-entropy loss. A cross-encoder (context + option in one sequence) is more accurate for reasoning such as negation but cannot cache option vectors; it is a valid alternative if bi-encoder accuracy is insufficient.
