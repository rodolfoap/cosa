# Formal Description: Latent Sequence Prediction Architecture (JEPA-Style Attention)

## 1. Overview
This document outlines the formal architecture and workflow for an experimental sequence-to-representation model. The system processes discrete input text tokens, converts them into continuous embeddings, applies a causal self-attention mechanism to build a sequential representation of context, predicts the subsequent representation in latent space, and projects that latent prediction back to a probability distribution over a discrete vocabulary.

---

## 2. Mathematical Formalism

### 2.1 Vocabulary and Inputs
Let $V = \{w_1, w_2, \dots, w_{|V|}\}$ be a finite vocabulary of size $|V|$.

An input sequence of length $T$ is represented as a sequence of discrete token indices:
$$\mathbf{X} = [x_1, x_2, \dots, x_T], \quad x_t \in \{1, \dots, |V|\}$$

Target token for next-token prediction:
$$y \in \{1, \dots, |V|\}$$

---

### 2.2 Embedding Mapping
An embedding layer $E: \{1, \dots, |V|\} \to \mathbb{R}^d$ maps each token index to a continuous vector of dimension $d$:
$$\mathbf{E}(\mathbf{X}) = [\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_T] \in \mathbb{R}^{T \times d}$$
where $\mathbf{e}_t = E(x_t) \in \mathbb{R}^d$.

---

### 2.3 Sequential Representation via Attention
To aggregate context while preserving autoregressive properties, a **Causal Scaled Dot-Product Multi-Head Attention** mechanism is applied:

$$\mathbf{Q} = \mathbf{E}(\mathbf{X})\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{E}(\mathbf{X})\mathbf{W}_K, \quad \mathbf{V} = \mathbf{E}(\mathbf{X})\mathbf{W}_V$$

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left( \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M} \right)\mathbf{V}$$

Where $\mathbf{M} \in \mathbb{R}^{T \times T}$ is a causal mask ensuring position $t$ only attends to positions $\le t$:
$$M_{i, j} = \begin{cases} 0 & \text{if } i \ge j \\ -\infty & \text{if } i < j \end{cases}$$

The output sequence of contextual representations is denoted as:
$$\mathbf{H} = [\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_T] \in \mathbb{R}^{T \times d}$$

The aggregated state representing the entire sequence up to step $T$ is the final hidden state:
$$\mathbf{z}_{context} = \mathbf{h}_T \in \mathbb{R}^d$$

---

### 2.4 Latent Representation Predictor
A predictive function $f_\theta: \mathbb{R}^d \to \mathbb{R}^d$ projects the contextual representation to predict the continuous vector representation of the next step:
$$\hat{\mathbf{z}}_{next} = f_\theta(\mathbf{z}_{context}) \in \mathbb{R}^d$$

---

### 2.5 Probability Decoding & Objective Function
To evaluate the predicted latent vector against target tokens and prevent representation collapse during training, $\hat{\mathbf{z}}_{next}$ is projected to token logits using a linear projection layer $\mathbf{W}_{out} \in \mathbb{R}^{|V| \times d}$:

$$\mathbf{z}_{logits} = \mathbf{W}_{out} \hat{\mathbf{z}}_{next} \in \mathbb{R}^{|V|}$$

The probability distribution over the vocabulary $P(y \mid \mathbf{X})$ is computed via Softmax:
$$P(y = k \mid \mathbf{X}) = \frac{\exp(z_{logits, k})}{\sum_{j=1}^{|V|} \exp(z_{logits, j})}$$

The training objective optimizes model parameters $\Theta$ using Cross-Entropy Loss against target label $y$:
$$\mathcal{L}(\Theta) = -\log P(y \mid \mathbf{X})$$

---

## 3. Concrete Example Execution

### Vocabulary Setup
Let $|V| = 6$:
$$V = \{ \text{"<PAD>"}, \text{"cat"}, \text{"sat"}, \text{"on"}, \text{"the"}, \text{"mat"} \}$$

Index mapping:
- `"cat"` $\to 1$
- `"sat"` $\to 2$
- `"on"`  $\to 3$
- `"the"` $\to 4$
- `"mat"` $\to 5$

---

### Data Sample
- **Input Sequence ($\mathbf{X}$):** `["cat", "sat", "on", "the"]` $\implies [1, 2, 3, 4]$ ($T = 4$)
- **Target Label ($y$):** `"mat"` $\implies 5$

---

### Step-by-Step Data Flow

```
Input Tokens [1, 2, 3, 4]
       │
       ▼
[Embedding Layer: E(X)]
       │
       ▼  (Dimensions: 4 x d)
[Causal Multi-Head Self-Attention]
       │
       ▼  (Sequence representation H: 4 x d)
[Extract Last Context Vector h_4]
       │
       ▼  (Vector z_context: 1 x d)
[Latent Predictor Neural Network f_theta]
       │
       ▼  (Predicted Vector z_next: 1 x d)
[Linear Decoding Layer: W_out]
       │
       ▼  (Logits: 1 x 6)
[Softmax Function]
       │
       ▼
[Probability Distribution over V]  ---> Target: Index 5 ("mat")
```

---

## 4. Key Considerations for Discussion with Your Teacher

1. **Latent Prediction vs. Generative Auto-regression:**
   - Standard Large Language Models (LLMs) output probabilities over tokens directly from the transformer layers.
   - This model introduces an explicit intermediate step: generating a predicted **latent vector** ($\hat{\mathbf{z}}_{next}$) before projecting back to vocabulary space.

2. **Mitigating Representation Collapse:**
   - Pure vector-to-vector loss functions (e.g., Mean Squared Error between $\hat{\mathbf{z}}_{next}$ and $E(y)$) often result in collapse where all predictions decay to a constant vector.
   - The current setup avoids collapse by using Cross-Entropy Loss over token probabilities. Alternative JEPA methods use VICReg or Normalized Temperature-scaled Cross Entropy (NT-Xent) contrastive loss.
