# RUDDER: Residual-Update Directed DEcoding Regulation

Official implementation of **RUDDER: Residual-Update Directed DEcoding Regulation**, an inference-time intervention method for mitigating object hallucination in Large Vision-Language Models (LVLMs).

RUDDER extracts an input-specific **Contextual Activation Residual Direction (CARD)** from the mandatory prefill pass and injects it into the residual stream during decoding. The injection is controlled by a token-level **Beta Gate**, which adaptively regulates the steering strength based on the alignment between the current hidden state and the visual evidence direction.

## Paper

**Adaptive Residual-Update Steering for Low-Overhead Hallucination Mitigation in Large Vision Language Models**  
Zhengtao Zou, Ya Gao, Jiarui Guan, Bin Li, Pekka Marttinen  
ICML 2026

## Method Overview

RUDDER is designed to reduce hallucination without adding extra forward passes.

The method has two main components:

1. **CARD vector**  
   A per-sample visual evidence direction extracted from self-attention residual updates during the prefill stage.

2. **Beta Gate**  
   A Bayesian-inspired adaptive gate that controls token-level steering strength during autoregressive decoding.

RUDDER is training-free, model-agnostic in principle, and implemented using lightweight forward hooks.

## Repository Structure

```text
.
├── configs/
│   └── config.py                  # Dataset paths and experiment hyperparameters
├── scripts/
│   ├── run_llava_chair.py          # CHAIR evaluation for LLaVA-1.5
│   ├── run_idefics2_chair.py       # CHAIR evaluation for Idefics2
│   ├── run_instructblip_chair.py   # CHAIR evaluation for InstructBLIP
│   └── run_qwen_chair.py           # CHAIR evaluation for Qwen2.5-VL
├── rudder/
│   ├── llava_methods.py            # RUDDER hooks and utilities for LLaVA
│   ├── idefics2_methods.py         # RUDDER hooks and utilities for Idefics2
│   ├── instructblip_methods.py     # RUDDER hooks and utilities for InstructBLIP
│   └── qwen_methods.py             # RUDDER hooks and utilities for Qwen2.5-VL
├── environment.yml
└── README.md
