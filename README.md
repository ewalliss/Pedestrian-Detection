# V-Track — Pedestrian Multi-Camera Re-Identification

Fine-tuned **CLIP ViT-B/16** for person re-identification across multiple cameras.
Built on [CLIP-ReID](original_CLIP-REID/CLIP-ReID) with the strongest variant: **ViT-CLIP-ReID-SIE-OLP**.

## Repository Note

This repository is a public demo clone of the private `pedestrian_detection` repo.
It exists for demo purposes and may omit internal assets, history, or tooling from the private source.

---

## Results on Market-1501

| Model | mAP | Rank-1 | Rank-5 | Rank-10 | Rank-1+RR | Params | Speed |
|-------|-----|--------|--------|---------|-----------|--------|-------|
| **Baseline** (CLIP-ReID, 60 ep) | 91.5% | 96.4% | 99.1% | 99.5% | 96.7% | 126.8M | 18.8 ms/img |
| **Custom** (ViT-CLIP-ReID-SIE-OLP, S1=3ep S2=60ep) | 93.0% | 97.2% | 99.2% | 99.5% | 97.7% | 127.2M | 21.2 ms/img |

> RR = k-reciprocal re-ranking.

---

## Retrieval Visualisations

Blue border = query. **Green** = correct match. **Red** = incorrect. Similarity scores shown below each result.

### query_005 — pid 633 · All top-5 correct
![query_005](./viz/retrieval_viz/query_005_pid633.jpg)

Distinctive bright yellow-green shirt: all 5 retrievals correct (sim 0.89–0.91) across different camera angles and poses.

---

### query_006 — pid 626 · All top-5 correct
![query_006](./viz/retrieval_viz/query_006_pid626.jpg)

Teal jacket + backpack: perfect top-5 retrieval (sim 0.91–0.93) including viewpoint changes.

---

### query_001 — pid 478 · Rank-1 correct (3/5)
![query_001](./viz/retrieval_viz/query_001_pid478.jpg)

Green shirt + dark shorts. Rank-1 and Rank-2 correct (sim 0.89–0.92). Two false positives at sim 0.88 — a different person with near-identical clothing, illustrating the hard-negative failure mode.

---

### query_003 — pid 570 · Rank-1 correct (3/5)
![query_003](./viz/retrieval_viz/query_003_pid570.jpg)

Black graphic T-shirt + dark shorts + green shoes. Top-3 correct (sim 0.82–0.90). Rank-4/5 false positives share the same outfit, sim drops to 0.69–0.77.

---

### query_019 — pid 461 · Rank-1 correct (3/5)
![query_019](./viz/retrieval_viz/query_019_pid461.jpg)

Dark top + white skirt + handbag. Top-3 correct (sim 0.80). Rank-4/5 false positives at sim 0.61 — large similarity gap separates true matches from false ones.

---

### query_002 — pid 18 · Rank-1 incorrect (4/5)
![query_002](./viz/retrieval_viz/query_002_pid18.jpg)

Back-view bicycle rider. Top-1 wrong (sim 0.78, different person on bicycle). Rank-2 through Rank-5 correct (sim 0.73–0.74). Failure driven by viewpoint ambiguity and lack of distinguishing front-facing features.

---

## Full Retrieval Gallery

All retrieval visualisation classes/queries currently available in the demo set:

### query_004 — pid 299
![query_004](./viz/retrieval_viz/query_004_pid299.jpg)

### query_007 — pid 339
![query_007](./viz/retrieval_viz/query_007_pid339.jpg)

### query_008 — pid 437
![query_008](./viz/retrieval_viz/query_008_pid437.jpg)

### query_009 — pid 100
![query_009](./viz/retrieval_viz/query_009_pid100.jpg)

### query_010 — pid 675
![query_010](./viz/retrieval_viz/query_010_pid675.jpg)

### query_011 — pid 291
![query_011](./viz/retrieval_viz/query_011_pid291.jpg)

### query_012 — pid 334
![query_012](./viz/retrieval_viz/query_012_pid334.jpg)

### query_013 — pid 61
![query_013](./viz/retrieval_viz/query_013_pid61.jpg)

### query_014 — pid 412
![query_014](./viz/retrieval_viz/query_014_pid412.jpg)

### query_015 — pid 334
![query_015](./viz/retrieval_viz/query_015_pid334.jpg)

### query_016 — pid 302
![query_016](./viz/retrieval_viz/query_016_pid302.jpg)

### query_017 — pid 610
![query_017](./viz/retrieval_viz/query_017_pid610.jpg)

### query_018 — pid 62
![query_018](./viz/retrieval_viz/query_018_pid62.jpg)

### query_020 — pid 138
![query_020](./viz/retrieval_viz/query_020_pid138.jpg)

---

## Model Architecture

Two-stage fine-tuning of CLIP ViT-B/16 with a split image/text training flow:

```
Input (224×224)
    ↓
ViT-B/16  [12 layers · 12 heads · 768-d]
    ├─ CLS token → visual_proj (768→512) → L2-norm                 [global feat]
    └─ patch tokens → SIELayer(cam/view embeddings)                [conditioned patches]
                         ↓
                   OLPHead(top-k by L2 norm → fuse → L2-norm)      [fused feat]
                         ↓
                 BN(512) → Linear(512 → num_pids)                  [Stage 2 classifier]
```

| Component | Role |
|-----------|------|
| **SIE** | Side Information Embedding — adds camera/view embeddings to the patch-token path |
| **OLP** | Overlapping Local Patches — selects top-k patch tokens by L2 norm and fuses them into the image representation |
| **PromptLearner** | Per-identity learnable text context used to build prompt embeddings for Stage 1 supervision |

### Training Protocol

| | Stage 1 | Stage 2 |
|-|---------|---------|
| **Trainable** | PromptLearner only | Image encoder + SIE + OLP + BN + classifier |
| **Frozen** | Image encoder + text encoder | Text-side supervision path remains frozen while image-side components are fine-tuned |
| **Loss** | Bidirectional supervised contrastive loss (I→T + T→I) | Weighted ID + Circle + I2T losses |
| **Purpose** | Learn identity-conditioned text prompts with frozen CLIP encoders | Refine image embeddings against frozen text supervision and identity classification |

Training weights and core hyperparameters are config-driven (`src/config/defaults.py`).

---

## Directory Structure

```
src/
├── config/      PedestrianReIDConfig — all hyperparameters
├── datasets/    Market1501, transforms, RandomIdentitySampler
├── models/      CLIPReIDPedestrianModel, PromptLearner, SIELayer, OLPHead
├── losses/      TripletLoss, IDLoss, SupConLoss
├── train/       train_stage1.py, train_stage2.py
├── reid/        ReIDPipeline, CLIPEncoder, CosineMatcher, TemporalFeatureBank
└── eval/        evaluate.py, reranking.py, model_health_check.py

scripts/
├── baseline_eval.py    Evaluate baseline + custom side-by-side
├── run_experiment.py   Full comparison → dated markdown report
└── mot_inference.py    Multi-camera MOT inference

model/
├── model_baseline/     Official CLIP-ReID checkpoint + metrics cache
└── custom/             Fine-tuned stage2_best.pth

docs/
├── viz/retrieval_viz/  Retrieval visualisation images
└── 2026-03-04-experiment-results.md
```

---

## Setup

```bash
# Python 3.13 + CUDA 11.8
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Evaluate

```bash
# Baseline + custom side-by-side comparison table
python scripts/baseline_eval.py \
    --market1501-root /path/to/Market-1501-v15.09.15

# Full dated markdown report
python scripts/run_experiment.py \
    --market1501-root /path/to/Market-1501-v15.09.15 \
    --skip-baseline                # reuse cached baseline JSON
```

---

## Train

```bash
# Stage 1 — PromptLearner (recommend ≥30 epochs)
python -m src.train.train_stage1

# Stage 2 — image encoder fine-tuning
python -m src.train.train_stage2

# Smoke test (3+2 epochs, 100 IDs, CPU-safe)
python -m src.train.train_stage1 --mini
python -m src.train.train_stage2 --mini
```

Key config fields (`src/config/defaults.py`):

| Field | Default | Note |
|-------|---------|------|
| `epochs_stage1` | 120 | Reduce to 30 for a faster re-train |
| `epochs_stage2` | 60 | |
| `match_threshold` | 0.7 | CosineMatcher min similarity |
| `olp_k` | 16 | Top-k patches for OLP fusion |
| `rerank_k1 / k2` | 20 / 6 | k-reciprocal reranking |

---

## Citation

Built on [CLIP-ReID](https://github.com/Syliz517/CLIP-ReID) (Li et al., 2023).
