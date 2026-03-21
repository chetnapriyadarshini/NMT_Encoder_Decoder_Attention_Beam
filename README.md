# Neural Machine Translation: Encoder–Decoder with Attention and Beam Search

A Jupyter Notebook implementing a sequence-to-sequence Neural Machine Translation (NMT) system for **Hindi-to-English translation**, incorporating a GRU-based encoder–decoder architecture, Bahdanau attention, and beam search decoding.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Notebook Contents](#notebook-contents)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)
- [Contact](#contact)

---

## Overview

This project implements an attention-augmented sequence-to-sequence model for neural machine translation from Hindi to English. The core objective is to demonstrate how additive (Bahdanau) attention improves alignment between source and target sequences compared to vanilla encoder–decoder models, and how beam search decoding produces higher-quality translations than greedy decoding at inference time.

---

## Architecture

```
Source (Hindi) → Encoder (GRU) → Context Vector
                                        ↓
               Attention Mechanism (Bahdanau)
                                        ↓
                    Decoder (GRU) → Target (English)
                                        ↓
                              Beam Search Decoding
```

**Key Components:**

- **Encoder:** A GRU-based recurrent encoder that processes the source sequence token-by-token and produces a sequence of hidden states.
- **Bahdanau Attention:** A learned alignment mechanism that computes a context vector as a weighted sum of encoder hidden states, conditioned on the current decoder state.
- **Decoder:** A GRU-based recurrent decoder that generates the target sequence one token at a time, guided by the attention context vector.
- **Beam Search:** A heuristic search algorithm that maintains the top-k candidate sequences at each decoding step, improving translation quality over greedy decoding.

---

## Notebook Contents

| Section | Description |
|---|---|
| Data Loading & Preprocessing | Loading Hindi–English parallel corpus, Unicode normalisation, tokenisation |
| Vocabulary Construction | Building source and target vocabulary with special tokens (`<sos>`, `<eos>`, `<pad>`) |
| Model Definition | Encoder, Attention, and Decoder class implementations in TensorFlow/Keras |
| Training | Teacher-forcing training loop with loss computation |
| Beam Search Decoding | Implementation of beam search for inference |
| Translation Results | Qualitative evaluation on sample sentences |
| Attention Visualisation | Heatmap of attention weights across source–target token pairs |

---

## Technologies Used

| Library | Purpose |
|---|---|
| `TensorFlow` / `Keras` | Model definition and training |
| `pandas` | Data handling |
| `matplotlib` | Attention visualisation and training plots |
| `unicodedata` | Unicode normalisation for Devanagari script |
| `scikit-learn` | Data splitting utilities |

---

## Setup and Installation

```bash
git clone https://github.com/chetnapriyadarshini/NMT_Encoder_Decoder_Attention_Beam.git
cd NMT_Encoder_Decoder_Attention_Beam
pip install tensorflow pandas matplotlib scikit-learn
```

Launch the notebook:

```bash
jupyter notebook nmt_attention.ipynb
```

> **Note:** A GPU runtime is strongly recommended for training. Google Colab or a local CUDA-enabled environment is suitable.

---

## Usage

Execute the notebook cells sequentially. The notebook is self-contained and includes markdown explanations at each stage. The attention heatmaps generated at inference time provide interpretable insight into which source tokens the decoder attends to when generating each target token.

---

## Results

The model produces grammatically plausible English translations for Hindi input sentences. Beam search consistently yields more fluent translations compared to greedy decoding. Attention weight visualisations confirm that the model learns meaningful source–target alignments.

---

## References

- Bahdanau, D., Cho, K., & Bengio, Y. (2014). *Neural Machine Translation by Jointly Learning to Align and Translate*. arXiv:1409.0473.
- Cho, K. et al. (2014). *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation*. arXiv:1406.1078.
- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). *Sequence to Sequence Learning with Neural Networks*. NeurIPS 2014.

---

## Contact

Created by [@chetnapriyadarshini](https://github.com/chetnapriyadarshini) — feel free to reach out with questions or suggestions.
