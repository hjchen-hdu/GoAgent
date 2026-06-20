# GoAgent

Official implementation of "GoAgent: Group-of-Agents Communication Topology Generation for LLM-based Multi-Agent Systems"

## Project Structure

```
GoAgent/
├── experiment/
│   ├── model.py              # Core GoAgent algorithm implementation
│   ├── gsm8k/               # GSM8K experiments
│   ├── mmlu/                # MMLU experiments
│   └── utils.py             # Experiment utilities
├── mas_framework/               # Utility tools (refers to GDesigner)
│   ├── agents/             # Agent implementations
│   ├── graph/              # Graph structure utilities
│   ├── llm/                # Language model interfaces
│   ├── tools/              # Coding, search, and other tools
│   └── utils/              # Helper utilities
└── datasets/               # Dataset storage
```

## Quick Start

### Add API keys in `template.env` and change its name to `.env`

```bash
BASE_URL = ""  # the BASE_URL of OpenAI LLM backend
API_KEY = ""   # for OpenAI LLM backend
```

### Run GoAgent on MMLU

#### MMLU (Knowledge Reasoning)

```bash
cd experiment/mmlu

# Cold start data generation
python cold_start_mmlu.py --dataset_name mmlu --llm_name gpt-4o-mini --batch_size 4 --num_iterations 10

# Training
python train_mmlu.py --pretrain --dataset_name mmlu --llm_name gpt-4o-mini --batch_size 4 --num_iterations 10  

# Evaluation
python evaluate_mmlu.py --model_path ./output/your_model_path --dataset_name mmlu --llm_name gpt-4o-mini --eval_batch_size 32
```

## Acknowledgments

This code refers to [GPTSwarm](https://github.com/metauto-ai/GPTSwarm), [GDesigner](https://github.com/yanweiyue/GDesigner), and [ARG-Designer](https://github.com/Shiy-Li/ARG-Designer).
