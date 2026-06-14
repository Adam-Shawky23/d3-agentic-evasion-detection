# Detecting Evasive Answers in Political Interviews — Fine-Tuned Transformers vs. Multi-Agent LLM Reasoning

> **TL;DR**: A 3-class NLP classification system that detects whether a politician's answer to a journalist's question is a *Clear Reply*, *Ambivalent*, or a *Clear Non-Reply*. I benchmark fine-tuned transformer encoders (BERT, DistilBERT, DeBERTa-v3) against a small open-weight LLM (Qwen3.5-0.8B) used both zero-shot and inside a custom **4-agent reasoning pipeline (D3-Agentic)**, then run a full ablation and error analysis to understand *why* each approach succeeds or fails.

---

## 🔍 Problem Statement

Political press conferences are full of deflection. Given a `(question, answer)` pair from an interview, the goal is to classify the **clarity of the response** into one of three categories:

| Label | Meaning |
|---|---|
| **Clear Reply** | The answer directly and fully addresses the question. |
| **Ambivalent** | The answer partially addresses it, is vague, or deflects without committing. |
| **Clear Non-Reply** | The answer ignores the question entirely or redirects to something else. |

This is a hard task: the dataset is **imbalanced** (Ambivalent ~59%, Clear Reply ~30%, Clear Non-Reply ~10%) and requires subtle pragmatic/linguistic reasoning rather than simple keyword matching.

---

## 📊 Dataset

- **Source**: [`ailsntua/QEvasion`](https://huggingface.co/datasets/ailsntua/QEvasion) (HuggingFace Hub)
- **Format**: `(question, answer, clarity_label)` triples from real political interview transcripts
- **Split**: Stratified 80/20 train/validation split (seed=42), plus a held-out competition test set (308 rows)
- Class weights were computed to counter the label imbalance during training

---

## 🧠 Approaches Implemented

This project doesn't just train one model — it builds a small **research benchmark** comparing five different systems:

### 1. Fine-tuned Encoder Baselines
- **BERT** and **DistilBERT** — standard fine-tuning with AdamW, linear warmup, weighted cross-entropy loss, and early stopping on Macro F1
- **DeBERTa-v3-base** — longer training schedule (20 epochs) with gradient accumulation, trained in float32 to avoid known fp16 instability issues

### 2. Zero-Shot LLM Baseline
- A single-prompt classifier using **Qwen3.5-0.8B**, asked directly to label each `(question, answer)` pair with no reasoning scaffold

### 3. D3-Agentic Prompting (custom multi-agent pipeline)
A 4-agent decomposition built from scratch, where each agent has a focused sub-task:

| Agent | Role |
|---|---|
| **Agent 1 — Question Intent** | Identifies what information the question is actually requesting |
| **Agent 2 — Answer Content** | Extracts what the answer actually covers (and what it omits) |
| **Agent 3 — Gap & Evasion Analysis** | Compares required vs. actual content to surface evasions |
| **Agent 4 — Decision** | Makes the final classification using all prior analyses as context |

### 4. Prompt Optimization with DSPy
Used `DSPy`'s `BootstrapFewShot` optimizer to automatically select few-shot demonstrations for the classification pipeline — an automated alternative to hand-crafted prompt engineering.

### 5. Agent Ablation Study
Systematically removed individual agents (Agent 1, 2, or 3) from the D3 pipeline across 5 configurations to measure each agent's individual contribution to final accuracy.

---

## 🏆 Results

| System | Accuracy | Macro F1 |
|---|---|---|
| BERT (fine-tuned) | 0.7 | 0.62 |
| DistilBERT (fine-tuned) | 0.68 | 0.55 |
| **DeBERTa-v3-base (fine-tuned)** | 0.70 | **0.6345** 🥇 |
| Zero-Shot Qwen3.5-0.8B | 0.26 | 0.173 |
| D3-Agentic Qwen3.5-0.8B | 0.5159 | 0.3373 |

> 💡 Run the notebook to populate the BERT / DistilBERT numbers — they're printed in the "Full System Comparison" output and saved to `full_comparison.png`.

### Key Findings

- **Fine-tuned encoders win.** DeBERTa-v3-base outperforms every LLM-prompting approach by a wide margin — the task depends on subtle linguistic cues that benefit from gradient-based adaptation rather than zero-shot reasoning in a 0.8B parameter model.
- **Agentic decomposition matters.** D3-Agentic (Macro F1 = 0.337) nearly **doubles** the Zero-Shot baseline (Macro F1 = 0.173). Breaking the task into 4 focused reasoning steps gives a small LLM enough structure to make meaningful predictions.
- **"Clear Non-Reply" is the hardest class** across every system. The Zero-Shot model collapses almost entirely toward predicting "Clear Non-Reply," while D3-Agentic over-predicts "Ambivalent" — the encoders are far more balanced across all three classes.
- **Failure modes are diagnosable.** Confusion matrices and per-class F1 breakdowns (see `confusion_matrices.png` and `per_class_f1.png`) make it clear *where* each system breaks down, not just *that* it does.

---

## 🛠️ Tech Stack

- **Python**, **PyTorch**, **HuggingFace Transformers & Datasets**
- **Models**: BERT, DistilBERT, DeBERTa-v3-base, Qwen3.5-0.8B
- **DSPy** for automated prompt optimization
- **Scikit-learn** for metrics (Macro F1, confusion matrices, classification reports)
- **Matplotlib / Seaborn** for visualizations
- Trained on Kaggle's T4 GPU environment

---

## 📁 Repository Structure

```
political-evasion-classifier/
├── README.md
├── notebook.ipynb              # Full pipeline: data loading → training → agentic inference → analysis
├── requirements.txt
└── results/
    ├── full_comparison.png         # Bar chart: all systems, Macro F1 & Accuracy
    ├── confusion_matrices.png       # Side-by-side confusion matrices for all 5 systems
    └── per_class_f1.png             # Per-class F1 comparison across systems
```

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

**Notes:**
- The first cell installs a development build of `transformers` (required for the `qwen3_5` model architecture) — **restart the kernel after running it**.
- The D3-Agentic full inference pass takes ~3.5 hours on a T4 GPU (690 validation + 308 test examples × 4 agent calls each). For a quick test, run the 2-example sanity check in the agent definition cell first.
- Requires GPU access (designed for ~16GB VRAM, e.g. Kaggle's T4).

---

## 📈 Possible Extensions

- Fine-tune the LLM (e.g. LoRA on Qwen3.5) instead of using it zero-shot/agentically
- Expand DSPy optimization beyond 12 training examples for a fairer comparison
- Try larger open-weight LLMs to see whether the agentic gap narrows
- Ensemble the best encoder (DeBERTa) with the D3-Agentic predictions

---

## 🙏 Acknowledgments

Dataset: [`ailsntua/QEvasion`](https://huggingface.co/datasets/ailsntua/QEvasion) on HuggingFace Hub. This project was developed as part of an NLP/LLM coursework assignment and extended into a full multi-system benchmark for portfolio purposes.
