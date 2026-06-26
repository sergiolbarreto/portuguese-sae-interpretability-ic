# Reproducibility Code: Portuguese Morphosyntactic Features with Sparse Autoencoders

**Anonymous submission for double-blind review.**

This repository contains the code, notebooks, directed probes, and raw result
JSONs to reproduce the experiments reported in the manuscript *"Discovering
and Steering Portuguese Morphosyntactic Features with Sparse Autoencoders"*.

> Author and affiliation withheld for double-blind review. Code will be
> attributed to the authors upon acceptance.

## Contents

```
notebooks/          # End-to-end pipeline (run in Colab or locally w/ GPU)
results/            # Raw JSON outputs behind every table/figure in the paper
data/fase5/         # LLM-generated feature descriptions + validation rubric
validation_study/   # Human-in-the-loop annotation results (sanitized)
scripts/            # Standalone analysis scripts
requirements.txt    # Python dependencies
```

## Notebooks

| Notebook | Phase | Reproduces |
|---|---|---|
| `fase1_fase2_piloto.ipynb` | 1–2 | Pilot analysis (1M tokens) |
| `fase3_analise_completa.ipynb` | 3 | Full-scale LSI + two-level triangulation (20M tokens, 4 corpora) |
| `fase4_probes.ipynb` | 4 | Directed linguistic probes (~170 texts, 6 phenomena) |
| `fase5_validacao.ipynb` | 5 | Human-in-the-loop validation (30 features, 9 annotators) |
| `fase6_steering.ipynb` | 6 | Feature steering: ablation + amplification |

### Extended experiments (Gemma 2 2B / 9B)

| Notebook | Reproduces |
|---|---|
| `exp_controle_espanhol.ipynb` | Spanish control → Romance hierarchy |
| `exp_probes_multilayer.ipynb` | Probes + causal benchmark at L9/L13/L17 |
| `exp_gender_benchmark.ipynb` | 228-item gender benchmark |
| `exp_multi_layer.ipynb` | Multi-layer LSI (layers 5, 9, 13, 17, 21) |
| `exp_random_ablation_control.ipynb` | Norm-matched random ablation controls |
| `exp_gemma9b_multifeature_ablation.ipynb` | 9B multi-feature ablation (K=1…20 collapse) |
| `exp_gemma9b_steering.ipynb` | 9B steering pipeline |
| `exp_gemma9b_lsi.ipynb` | 9B LSI |
| `exp_gemma9b_probes.ipynb` | 9B probes |
| `exp_amplificacao_causal.ipynb` | Causal amplification (crase, clitics, up to 16×) |
| `exp_sae_65k.ipynb` | Cross-scale SAE validation (65k features) |
| `exp_robustez_layer13.ipynb` | Multi-seed robustness, bootstrap CIs |
| `exp_register_steering_v2.ipynb` | Register steering (n=50 prompts) |

## Setup

```bash
pip install -r requirements.txt
```

Dependencies include `transformers`, `sae_lens`, `transformer_lens`, `torch`,
`datasets`, `krippendorff`, `numpy`, `pandas`, `matplotlib`.

## Hardware

- **Gemma 2 2B**: NVIDIA T4 (16 GB VRAM) or equivalent.
- **Gemma 2 9B**: NVIDIA A100 (40 GB) or equivalent.

## Models and SAEs

- Base model: **Gemma 2 2B** and **Gemma 2 9B** (Google DeepMind).
- Sparse autoencoders: **Gemma Scope** canonical SAEs (JumpReLU, 16k and 65k
  expansions) via SAELens.

## Corpora

Two-level triangulation across four corpora (5M tokens each, 20M total):
- **Web-crawl**: FineWeb-2 PT vs. FineWeb EN
- **Wikipedia**: Wikipedia PT vs. Wikipedia EN
- **Spanish control**: Spanish Wikipedia + FineWeb-2 Spanish (1M tokens each)

> Checkpoint activations are large and not shipped here; the notebooks
> regenerate them from the corpora. Adjust `CHECKPOINT_DIR` / `SAVE_DIR` paths
> at the top of each notebook to point to your local or Colab storage.

## Validation study (sanitized)

`validation_study/iaa_results/` contains the inter-annotator agreement
results. Annotator-identifying columns (name, course, individual identifiers)
have been removed; annotators are reported only by anonymized codes. The
author-anchor column is labeled `kappa_vs_anchor`.

## Citation

```
@misc{anonymous2026portuguese,
  title  = {Discovering and Steering Portuguese Morphosyntactic Features
            with Sparse Autoencoders},
  author = {Anonymous},
  year   = {2026},
  note   = {Submitted for double-blind review}
}
```

## License

See `LICENSE`.
