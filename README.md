# RUDDER: Residual-Update Directed DEcoding Regulation

[![Paper](https://img.shields.io/badge/arXiv-2511.10292-b31b1b.svg)](https://arxiv.org/abs/2511.10292)
[![Conference](https://img.shields.io/badge/ICML-2026-blue.svg)]()
[![Code](https://img.shields.io/badge/GitHub-RUDDER-black.svg)](https://github.com/Akko000/RUDDER-Residual-Update-Directed-DEcoding-Regulation-)

Official implementation of **RUDDER**, a low-overhead inference-time intervention method for mitigating object hallucination in Large Vision-Language Models (LVLMs).

> **Adaptive Residual-Update Steering for Low-Overhead Hallucination Mitigation in Large Vision Language Models**  
> Zhengtao Zou, Ya Gao, Jiarui Guan, Bin Li, Pekka Marttinen  
> ICML 2026  
> Paper: https://arxiv.org/abs/2511.10292

<p align="center">
  <img src="assets/teaser.pdf" width="90%">
</p>

## Overview

Large Vision-Language Models often hallucinate objects because visual information is encoded as an initial prefix and can become diluted during autoregressive decoding. As generation proceeds, the model may over-rely on language priors instead of the visual evidence.

**RUDDER** addresses this problem by extracting a per-sample visual evidence direction from the model's own residual updates during the prefill stage, then injecting this direction during decoding with adaptive token-wise control.

RUDDER is:

- **Training-free**: no model weight update is required.
- **Single-pass**: no extra forward pass is needed.
- **Adaptive**: injection strength is controlled per token.
- **Low-overhead**: maintains around vanilla-level decoding throughput.
- **General**: works across LLaVA-1.5, Idefics2, InstructBLIP, and Qwen2.5-VL.

## Method

RUDDER contains two main components.

### 1. CARD: Contextual Activation Residual Direction

During the prefill stage, RUDDER extracts a visual evidence vector from the self-attention residual updates of a selected decoder layer.

Given the residual update at layer `l` for token `i`:

```text
Delta_i^l = A_i^l
```

RUDDER pools these updates over the prefill span and normalizes the result:

```text
v_CARD = Pool({Delta_i^l}) / ||Pool({Delta_i^l})||_2
```

This produces an input-specific direction that acts as a persistent visual anchor.

### 2. Beta Gate

During decoding, RUDDER computes the cosine similarity between the current hidden state and `v_CARD`, then maps it to an adaptive gate using a Beta-inspired formulation:

```text
s_t = cos(h_{l,t}, v_CARD)

alpha_t = softplus(k * s_t + c)
beta_t  = softplus(-k * s_t + c)

g_t = alpha_t / (alpha_t + beta_t)
```

The final steering vector is:

```text
v_t^steer = alpha_max * g_t * v_CARD
```

This vector is injected into the residual stream only during answer generation.

<p align="center">
  <img src="assets/method_overview.pdf" width="90%">
</p>

## Main Results

RUDDER reduces hallucination while preserving general multimodal capability.

### CHAIR Results

| Model | Method | CHAIRs ↓ | CHAIRi ↓ |
|---|---|---:|---:|
| LLaVA-1.5-7B | Vanilla | 48.6 | 13.6 |
| LLaVA-1.5-7B | RUDDER-Beta | 39.5 | 10.5 |
| Idefics2-8B | Vanilla | 46.6 | 14.9 |
| Idefics2-8B | RUDDER-Beta | 28.4 | 10.9 |
| InstructBLIP | Vanilla | 39.2 | 12.8 |
| InstructBLIP | RUDDER-Beta | 27.1 | 8.5 |
| Qwen2.5-VL-7B | Vanilla | 35.2 | 9.5 |
| Qwen2.5-VL-7B | RUDDER | 26.9 | 7.0 |

### POPE Results

RUDDER also improves object-probing performance on POPE. On LLaVA-1.5, RUDDER-Beta improves greedy decoding accuracy from **85.34** to **86.53** and F1 from **84.91** to **86.03**.

### Efficiency

RUDDER operates within a single inference pass and introduces little overhead. In our experiments, **RUDDER-Beta maintains about 96% throughput** compared with vanilla decoding, while many contrastive or steering baselines require extra forward passes.

| Method | LLaVA-1.5 | Idefics2 | InstructBLIP |
|---|---:|---:|---:|
| Vanilla | 56.7 | 47.8 | 62.3 |
| VISTA | 36.1 | 31.9 | 28.9 |
| RUDDER-Beta | 54.9 | 45.8 | 59.5 |
| RUDDER-Add | 55.8 | 46.5 | 60.8 |

## Installation

```bash
git clone https://github.com/Akko000/RUDDER-Residual-Update-Directed-DEcoding-Regulation-.git
cd RUDDER-Residual-Update-Directed-DEcoding-Regulation-

conda create -n rudder python=3.10
conda activate rudder

pip install -r requirements.txt
```

## Usage

Example command:

```bash
python run_rudder.py \
  --model llava-1.5-7b \
  --image examples/demo.jpg \
  --prompt "Please help me describe the image in detail." \
  --method rudder_beta
```

A typical RUDDER inference pipeline contains:

```python
# 1. Load the LVLM backbone.
# 2. Register a hook at the selected decoder layer.
# 3. Extract CARD during the prefill stage.
# 4. Decode with RUDDER-Beta or RUDDER-Add.
```

## Assets

Recommended files under `assets/`:

```text
assets/
├── teaser.png              # Figure 1: hallucination example and method comparison
├── method_overview.png     # Figure 2: RUDDER workflow
├── chair_results.png       # Table 1: CHAIR main results, optional
├── efficiency.png          # Table 4: throughput comparison, optional
├── ablation_layer.png      # Figure 3: layer ablation, optional
└── hyperparam_heatmap.png  # Figure 4: alpha/k sensitivity, optional
```

For the README, the most important figures are:

1. `teaser.png`
2. `method_overview.png`

The other figures are optional and can be used if you want a more complete project page.

## Repository Structure

```text
RUDDER/
├── assets/                 # Figures used in README
├── examples/               # Example images and prompts
├── Idefics2/                 
│   ├── Ide_methods.py
│   ├── Ide_methods.py              
│   └── Ide_chair.py        
│   
├──               # Evaluation and demo scripts
├── environment.yml
└── README.md
```

## Citation

```bibtex
@inproceedings{zou2026rudder,
  title={Adaptive Residual-Update Steering for Low-Overhead Hallucination Mitigation in Large Vision Language Models},
  author={Zou, Zhengtao and Gao, Ya and Guan, Jiarui and Li, Bin and Marttinen, Pekka},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}
```

## Acknowledgements

This work was supported by the Research Council of Finland, including the Finnish Center for Artificial Intelligence FCAI Flagship programme and grant 358246.

## License

Please check the repository license before using or modifying the code.
