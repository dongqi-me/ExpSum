# 📚 Explanatory Summarization with Discourse-Driven Planning

This repository contains the official implementation of "Explanatory Summarization with Discourse-Driven Planning" accepted at Transactions of the Association for Computational Linguistics (TACL).

<p align="center">
  <img src="https://img.shields.io/badge/TACL-2025-blue" alt="TACL 2025">
  <img src="https://img.shields.io/badge/Status-Available-green" alt="Status: Available">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License: MIT">
</p>

## 📄 Paper Abstract

Lay summaries for scientific documents typically include explanations to help readers grasp sophisticated concepts or arguments. However, current automatic summarization methods do not explicitly model explanations, which makes it difficult to align the proportion of explanatory content with human-written summaries. In this paper, we present a plan-based approach that leverages discourse frameworks to organize summary generation and guide explanatory sentences by prompting responses to the plan. Specifically, we propose two discourse-driven planning strategies, where the plan is conditioned as part of the input or part of the output prefix, respectively. Empirical experiments on three lay summarization datasets show that our approach outperforms existing state-of-the-art methods in terms of summary quality, and it enhances model robustness, controllability, and mitigates hallucination.

## 🔧 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Training Plan-Output Model
```bash
python src/train.py --model_type plan_output --train_data_path data/train.json --val_data_path data/val.json --output_dir models/plan_output
```

### Training Plan-Input Model
```bash
python src/train.py --model_type plan_input --train_data_path data/train.json --val_data_path data/val.json --output_dir models/plan_input
```

### Inference
```bash
python src/inference.py --model_type plan_output --model_path models/plan_output --test_data_path data/test.json --output_path results.json


python src/inference.py --model_type plan_output --model_path models/plan_input --test_data_path data/test.json --output_path results.json
```

## 📚 Parser References

This work builds upon several discourse parsing frameworks. For readers interested in the underlying discourse parsing techniques, we recommend the following references:

1. Zhengyuan Liu, Ke Shi, and Nancy Chen. 2020. [Multilingual Neural RST Discourse Parsing](https://aclanthology.org/2020.coling-main.591/). In *Proceedings of the 28th International Conference on Computational Linguistics*, pages 6730–6738, Barcelona, Spain (Online). International Committee on Computational Linguistics.

2. Grigorii Guz and Giuseppe Carenini. 2020. [Coreference for Discourse Parsing: A Neural Approach](https://aclanthology.org/2020.codi-1.17/). In *Proceedings of the First Workshop on Computational Approaches to Discourse*, pages 160–167, Online. Association for Computational Linguistics.

3. Aru Maekawa, Tsutomu Hirao, Hidetaka Kamigaito, and Manabu Okumura. 2024. [Can we obtain significant success in RST discourse parsing by using Large Language Models?](https://aclanthology.org/2024.eacl-long.171/). In *Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 2803–2815, St. Julian’s, Malta. Association for Computational Linguistics.

## 📝 Citation

```bibtex
@article{liu2025explanatory,
  title={Explanatory Summarization with Discourse-Driven Planning},
  author={Liu, Dongqi and Yu, Xi and Demberg, Vera and Lapata, Mirella},
  journal={arXiv preprint arXiv:2504.19339},
  year={2025}
}
```
