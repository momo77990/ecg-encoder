# ECG Encoder

ECG Encoder is a two-stage ECG understanding project for MIT-BIH heartbeat classification and ECG-to-language diagnosis generation.

## Overview

The project contains two phases:

1. **ECG Transformer encoder**: a wav2vec 2.0 / ECG-FM style encoder trained from scratch on MIT-BIH heartbeat data for 5-class arrhythmia classification.
2. **ECG + LLM alignment**: a LLaVA-style stage-1 setup that freezes the ECG encoder and Qwen2.5-1.5B-Instruct, then trains an MLP projector so the LLM can generate natural-language ECG diagnoses.

Reported results from the project notes:

- Phase 1 ECG classifier macro F1: **0.8625**
- Phase 2 LLM generation + parsed classification macro F1: **0.9057**

## Model architecture

![ECG Encoder model architecture](assests/model.png)

The model is organized as a two-stage pipeline. In Phase 1, the MIT-BIH ECG signal is first encoded by a CNN + Transformer ECG encoder. The encoder converts each heartbeat record into sequence features, and a mean-pooling classification head predicts one of five heartbeat classes: N, S, V, F, or Q.

After Phase 1 training, the classification head is discarded. The trained ECG encoder is frozen and reused in Phase 2. Its ECG sequence features are passed through a trainable MLP projector, which maps ECG embeddings into the same hidden dimension as the language model token embeddings.

In Phase 2, the projected ECG features are treated as **ECG soft tokens** and inserted into a ChatML-style prompt. During training, the input is constructed as:

```text
Prefix prompt + ECG soft tokens + Suffix prompt + Target diagnosis text
```

The loss is calculated only on the target diagnosis text tokens. This means the model learns to generate the diagnosis from the prompt and ECG information, instead of learning to reproduce the prompt itself. The Qwen2.5-1.5B-Instruct language model remains frozen, and only the ECG-to-LLM projector is trained.

## Dataset

The project uses the MIT-BIH subset from the Kaggle heartbeat dataset:

- Source: `shayanfazeli/heartbeat`
- Files: `mitbih_train.csv`, `mitbih_test.csv`
- Signal format: single-lead ECG, 187 samples per record
- Labels: 5 classes `{0: N, 1: S, 2: V, 3: F, 4: Q}`

Data files are not included in this repository. Put them under `data/` before training or evaluation.

## Repository structure

```text
.
├── config.yaml              # Phase 1 ECG classifier config
├── config_lm.yaml           # Phase 2 ECG-LLM projector config
├── config_lm_lora.yaml      # LoRA-related config
├── requirements.txt
├── scripts/
│   └── extract_data.sh
└── src/
    ├── data/                # MIT-BIH dataset loader and caption templates
    ├── model/               # ECG encoder, projector, and ECG-LLM modules
    ├── train.py             # Phase 1 training
    ├── evaluate.py          # Phase 1 evaluation
    ├── train_lm.py          # Phase 2 projector training
    ├── evaluate_lm.py       # Phase 2 evaluation
    └── train_lm_lora.py     # LoRA fine-tuning
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train the ECG classifier

```bash
python -m src.train --config config.yaml
```

### Evaluate the ECG classifier

```bash
python -m src.evaluate --config config.yaml
```

### Train the ECG-LLM projector

```bash
python -m src.train_lm --config config_lm.yaml
```

### Evaluate the ECG-LLM model

```bash
python -m src.evaluate_lm --config config_lm.yaml
```

## Notes

Generated files, local data, training outputs, and helper shell scripts are excluded from Git by `.gitignore`.
