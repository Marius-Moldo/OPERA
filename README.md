<div align="center">
  <h1>CoughPhase-CLR</h1>
  <p><em>An acoustics-informed foundation model for coughing sound classification</em></p>
</div>

-----------------------------------------
[![Language: Python](https://img.shields.io/badge/language-Python%203.10%2B-green?logo=python&logoColor=green)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Built on: OPERA](https://img.shields.io/badge/built%20on-OPERA-blue.svg)](https://github.com/evelyn0414/OPERA)

This repository contains the code for **CoughPhase-CLR**, a self-supervised learning
framework that leverages the *physiological phases of a cough* for robust representation
learning. Unlike generic contrastive frameworks that build positive pairs through random
cropping, CoughPhase-CLR constructs positive pairs from the distinct acoustic phases of a
single cough event. The model is pretrained on ~40 hours of public cough audio and
evaluated on five downstream tasks spanning COVID-19 detection, COPD state classification,
gender, and smoker-status prediction.

This is the code release for the paper *"CoughPhase-CLR: Designing an acoustics-informed
foundation model for coughing sound classification"* (Moldovan et al., Chair of Health
Informatics, TUM). Key findings: cough-specific phase-aware pretraining consistently
outperforms standard random cropping when pretraining on coughs, and is more
**data-efficient**; however, COPD-state classification from coughs remains hard (best 57 %
UAR vs. 84 % from speech).

> **Built on OPERA.** This work builds on and reuses the pretraining and benchmarking
> scaffolding of [OPERA](https://github.com/evelyn0414/OPERA) (Zhang et al., 2024)

## Method

CoughPhase-CLR shares OPERA-CE's architecture (an **EfficientNet-B0** encoder trained with
a contrastive objective) but differs in how contrastive views are constructed:

- Each cough is split into **two segments**: the **explosive phase** (first segment) and
  the combined **intermediate + voiced phases** (second segment). The split point is found
  by detecting the largest energy peak in the first 60 % of the cough and cutting ~20 ms
  after it.
- The two segments of the *same* cough form a **positive pair**; segments from other coughs
  in the batch are negatives.
- Mel spectrograms (64 bins, window 1024, hop 512) are computed per segment, with
  SpecAugment applied during pretraining. Batch size 256.

For comparison, **OPERA-CE-Cough** is OPERA-CE re-pretrained on the *same* cough data but
using the original random-crop view construction. This isolates the effect of the
phase-aware contrastive task.

## Installation

The environment with all the needed dependencies can be created on a Linux machine by
running:

```
git clone <your-repo-url>/CoughPhase-CLR.git
cd ./CoughPhase-CLR

conda env create --file environment.yml
sh ./prepare_env.sh
source ~/.bashrc

conda init
conda activate audio
sh ./prepare_code.sh
```

This sets up a Python 3.10 environment (PyTorch 2.3, torchaudio, librosa, transformers,
PyTorch Lightning, timm). `prepare_env.sh` adds the project to `PYTHONPATH`; `prepare_code.sh`
patches the custom `swin_transformer.py` into the timm install. Afterwards you only need
`conda activate audio` to run the code.

## Datasets

CoughPhase-CLR is **pretrained** on two openly available cough datasets and **evaluated**
on four datasets (three public + one private).

| Dataset | Used for | Source | Access | License |
| ------- | -------- | ------ | ------ | ------- |
| UK COVID-19 | pretrain + eval | UKHSA / Alan Turing Inst. | [zenodo.org/records/10043978](https://zenodo.org/records/10043978) | OGL 3.0 |
| COUGHVID | pretrain + eval | EPFL | [zenodo.org/records/4048312](https://zenodo.org/records/4048312) | CC BY 4.0 |
| Coswara | eval | IISc | [github.com/iiscleap/Coswara-Data](https://github.com/iiscleap/Coswara-Data) | CC BY 4.0 |
| COPD-DE | eval | Univ. Hospital Augsburg | private — not publicly available | — |

*COVID-19 Sounds (used by OPERA for pretraining) was not available for this work due to
licensing. COPD-DE is a private clinical dataset of forced-cough recordings from COPD
patients (exacerbation vs. stable state) and is not redistributed here.*

## Data preparation

Pretraining uses a **segmented cough** pipeline: long recordings are resampled to 16 kHz,
individual coughs are extracted by energy-based onset/offset detection, each cough is split
into its two acoustic phases, and per-segment Mel spectrograms are saved.

Relevant scripts:

- `src/segment_cough.py`, `src/split_cough_final.py` — cough extraction and two-phase split.
- `src/pretrain/prepare_data/coughvid_pressl_segmented.py` — prepare segmented COUGHVID.
- `src/pretrain/prepare_data/covidUK_pressl_segmented.py` — prepare segmented UK COVID-19.
- `datasets/dataset_filename_utils.py` — shared helpers for building/transforming the
  `.npy` filename lists each dataset uses.

## Pretraining CoughPhase-CLR

The phase-aware model is trained with `src/pretrain/cola_training.py`:

```
python -u src/pretrain/cola_training.py \
  --title "CoughPhase-CLR-100pct" \
  --covidUKcough True --coughvid True \
  --encoder efficientnet \
  --preprocessing segmented \
  --strategy phase \
  --data_percentage 1.0 \
  --specaugment True \
  --target_steps 20000
```

Key flags:

- `--strategy phase` — phase-aware positive pairs (CoughPhase-CLR). The default
  `--strategy crop` reproduces the random-crop **OPERA-CE-Cough** baseline.
- `--preprocessing segmented` — use the segmented per-cough spectrograms.
- `--data_percentage` — fraction of pretraining data (for the data-efficiency study).
- `--target_steps`, `--specaugment`, `--encoder`, `--batch_size` — training controls.

To reproduce the full **data-efficiency sweep** (100/80/60/40/20 %) for both CoughPhase-CLR
and OPERA-CE-Cough:

```
sh scripts/multiple_pretrain.sh
```

Trained checkpoints are written under `cks/model/combined/coughvid_covidUKcough/`.

## Running the benchmark

CoughPhase-CLR is evaluated against baselines on five downstream tasks using a
**linear-probing** protocol (frozen encoder, linear head, AUROC over 5 seeds; UAR with
4-fold patient-level CV for COPD-DE). The benchmark scripts share
`scripts/lib/benchmark_common.sh`, which runs a feature-extraction phase followed by
`src/benchmark/linear_eval.py`.

| Task | Dataset | `linear_eval` task | Script |
| ---- | ------- | ------------------ | ------ |
| T1 | COUGHVID — COVID / non-COVID | `coughvidcovid` | `linear_eval --task coughvidcovid` (see `scripts/eval_all.sh`) |
| T2 | COUGHVID — female / male | `coughvidsex` | `scripts/coughvid_benchmark.sh` |
| T3 | COPD-DE — exacerbation / stable | `customCoughCOPD` | `scripts/custom_copd_benchmark.sh` |
| T4 | Coswara — smoker / non-smoker | `coswarasmoker` | `scripts/coswara_benchmark.sh` |
| T5 | Coswara — female / male | `coswarasex` | `scripts/data_efficiency_benchmark.sh` |

`scripts/data_efficiency_benchmark.sh` evaluates the 20–100 % checkpoints (eval-only; assumes
features are already extracted). For the public COPD dataset and full fine-tuning baselines,
see `scripts/copd_eval.sh` and `scripts/finetune_eval.sh`.

## Using a trained model

Feature extraction reuses OPERA's `extract_opera_feature`. The OPERA encoders
(`operaCT`, `operaCE`, `operaGT`) auto-download from HuggingFace on first use; locally
trained checkpoints (including CoughPhase-CLR) live under `cks/model/`.

```python
import numpy as np
from src.benchmark.model_util import extract_opera_feature

# array of filenames
sound_dir_loc = np.load(feature_dir + "sound_dir_loc.npy")
opera_features = extract_opera_feature(sound_dir_loc, pretrain="operaCE", input_sec=8, dim=1280)
np.save(feature_dir + "operaCE_feature.npy", np.array(opera_features))
```

Encoder dimensions: `operaCE` / CoughPhase-CLR (EfficientNet-B0) → 1280, `operaCT` → 768,
`operaGT` → 384.

## Repository layout

- `src/model/` — model definitions (EfficientNet/COLA, HT-SAT, MAE/ViT).
- `src/pretrain/` — pretraining entry points (`cola_training.py`, `mae_training.py`) and
  `prepare_data/` preprocessing.
- `src/benchmark/` — feature extraction (`model_util.py`), linear evaluation
  (`linear_eval.py`), per-dataset `processing/`, and baselines.
- `scripts/` — pretraining and benchmark scripts (+ shared `lib/benchmark_common.sh`).
- `datasets/` — datasets and dataset-specific preprocessing utilities.
- `cks/` — model checkpoints and logs.
- `res_analysis/` — result analysis and visualization.

## Citation

If you use this code, please cite **CoughPhase-CLR** and the **OPERA** framework it builds on.

```
@misc{moldovan2025coughphaseclr,
      title={CoughPhase-CLR: Designing an acoustics-informed foundation model for coughing sound classification},
      author={Marius Moldovan and Anton Batliner and Thomas M. Berghaus and Bj\"orn W. Schuller and Andreas Triantafyllopoulos},
      year={2025},
}
```

## License

Released under the MIT License
