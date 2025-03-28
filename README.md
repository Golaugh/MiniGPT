# MiniGPT: A Minimal Transformer Language Model

This is a from-scratch implementation of a GPT-style language model only for practice purpose.

---

## Project Structure

```text
MiniGPT/
├── config/
│   └── train_config.json        
├── data/
│   └── tokenizer.py             
├── model/
│   ├── embedding.py             
│   ├── gpt.py                   
│   └── transformer_block.py    
├── params/
│   └── gpt.pt                   
├── utils/
│   ├── logging.py               
│   └── metrics.py               
├── generate.py                  
└── train.py           
└── README.md                               
```

---

## Features

- Modular GPT components
- Config-driven training
- Supports custom datasets and tokenizers
- Easy extension practice using LoRA, quantization, and PEFT techniques

---

## Instructions

**Edit hyperparameters** in `config/train_config.json`

**Run training**:

```bash
    python train.py --config config/model_config.json
```
The **training loss** will then be saved in `log/stats_timestamp.jsonl`

And the **model params** will be saved after each console epoch in `params/gpt.pt`

**Run generation**:

```bash
    python generate.py --config config/model_config.json
```