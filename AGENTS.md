# AGENTS.md — FloodFrame

This file defines the rules and source-of-truth assumptions that AI coding agents must follow when working on FloodFrame.

---

## 1. Product Identity

FloodFrame is currently a **single image / camera-frame classifier**.

Verified task:

```text
Input  : one road/camera image
Output : FLOOD or NORMAL + confidence + class probabilities
```

Do not describe the current product as live CCTV monitoring.

---

## 2. Product Direction

Long-term direction:

```text
CCTV / Road Camera
        ↓
Video Stream
        ↓
Frame Sampling
        ↓
Flood Classification
        ↓
Temporal Persistence
        ↓
Camera Metadata / Location
        ↓
Alert / Dashboard
        ↓
Human Verification
```

Current implementation stops at:

```text
Image / Frame
→ EfficientNet-B0
→ FLOOD / NORMAL
→ Confidence
```

---

## 3. Current Model

Verified model:

```text
EfficientNet-B0
```

Preferred inference checkpoint:

```text
models/floodframe_efficientnet_b0.pth
```

Do not silently change architecture without updating:
- training code,
- inference code,
- README,
- evaluation reports,
- checkpoint compatibility.

---

## 4. Dataset Grounding

Flood source:

```text
FloodIMG
/kaggle/input/datasets/hhrclemson/flooding-image-dataset
```

Observed:

```text
9,296 flood images
```

Normal-road source:

```text
BDD100K
/kaggle/input/datasets/solesensei/solesensei_bdd100k
```

Observed:

```text
136,000 road images
```

Baseline used:

```text
6,000 FLOOD
6,000 NORMAL
12,000 total
```

Split:

```text
Train : 8,400
Val   : 1,800
Test  : 1,800
```

Never invent or alter these values unless a new verified experiment is run.

---

## 5. Known Dataset Risk

FLOOD and NORMAL come from different source datasets.

Known risk:

```text
dataset-source bias
```

Never claim that high test performance proves real-world generalization.

Always distinguish:

```text
experimental split performance
```

from:

```text
real-world CCTV performance
```

---

## 6. Verified Experimental Results

```text
Accuracy        ≈ 99.89%
Macro F1        ≈ 99.89%
Flood Precision = 100.00%
Flood Recall    ≈ 99.78%
Flood F1        ≈ 99.89%
```

Safe wording:

```text
FloodFrame reached approximately 99.89% accuracy on the current experiment test split.
```

Unsafe wording:

```text
FloodFrame is 99.89% accurate in the real world.
```

---

## 7. Confidence Rule

Critical rule:

```text
classification confidence ≠ flood severity
```

Never convert confidence into:
- flood depth,
- flood severity,
- water level,
- danger level,
- urgency level.

Example:

```text
FLOOD — 96%
```

means only that the model is highly confident the image belongs to the learned FLOOD class.

---

## 8. Single Frame ≠ Validated Flood Event

A single positive frame is not automatically a validated flood event.

Future event logic should consider:

```text
multiple consecutive positive frames
→ temporal persistence
→ operator review
→ alert
```

Do not implement production-style emergency alerts from one frame without explicit scope and warning labels.

---

## 9. Human-in-the-Loop

Preserve:

```text
AI gives indication.
Human gives decision.
```

Preferred UI wording:

```text
Indikasi banjir terdeteksi.
Perlu verifikasi operator.
```

Avoid:

```text
Banjir dipastikan terjadi.
```

---

## 10. Current Inference Flow

```text
Upload image
→ RGB conversion
→ Resize 224 × 224
→ Normalize
→ EfficientNet-B0
→ Softmax
→ FLOOD / NORMAL
→ confidence + class probabilities
```

Normalization:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Prefer checkpoint-provided normalization values when available.

---

## 11. Project Structure

Expected:

```text
floodframe/
├── assets/
├── data/
├── models/
├── notebooks/
├── reports/
├── src/
├── app.py
├── AGENTS.md
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

Primary files:

```text
app.py
src/predict.py
src/train.py
src/evaluate.py
notebooks/flood.ipynb
models/floodframe_efficientnet_b0.pth
reports/test_metrics.json
```

---

## 12. Model Path Rule

Preferred local inference model:

```text
models/floodframe_efficientnet_b0.pth
```

Do not default back to:

```text
runs/floodframe/best_model.pth
```

unless explicitly restoring an older checkpoint workflow.

---

## 13. Git Rules

Do not commit:
- raw datasets,
- Kaggle credentials,
- temporary files,
- local environments,
- caches,
- unrelated model checkpoints.

If committing the final model intentionally, use a specific `.gitignore` exception instead of unignoring all `.pth` files.

Example:

```gitignore
*.pth
!models/floodframe_efficientnet_b0.pth
```

---

## 14. Training Rules

Training should be reproducible.

Record:
- dataset sources,
- selected sample count,
- split strategy,
- seed,
- architecture,
- input size,
- optimizer,
- learning rate,
- epochs,
- evaluation metrics.

Never invent results.

---

## 15. Leakage Rules

Prevent:
- duplicate leakage,
- near-duplicate leakage,
- train/test contamination,
- same-event leakage.

For future video datasets, split by:

```text
video / camera / event
```

instead of randomly splitting individual frames.

---

## 16. External Validation

Before claiming stronger generalization, test:
- flood images outside FloodIMG,
- normal roads outside BDD100K,
- wet roads without flooding,
- puddles,
- reflections,
- night scenes,
- low-light CCTV,
- unusual camera angles.

Document failures as well as successes.

---

## 17. Roadmap Discipline

Do not jump from image classification directly to claims of real-time CCTV monitoring.

Expected evolution:

```text
v0.1 image/frame classification
v0.2 external validation
v0.3 video upload
v0.4 frame sampling
v0.5 temporal persistence
v0.6 camera metadata/location
v0.7 alert policy
v0.8 dashboard
v0.9 edge deployment
v1.0 integrated monitoring support
```

---

## 18. UI Rules

The Streamlit app should:
- show the uploaded frame,
- show the predicted class,
- show confidence,
- show class probabilities,
- explain confidence is not severity,
- state that operator verification is required.

Do not show fake metrics or unsupported operational claims.

---

## 19. Engineering Rules for AI Agents

Before modifying code:

1. Read `README.md`.
2. Read `AGENTS.md`.
3. Inspect current implementation.
4. Preserve working behavior unless a change is requested.
5. Define the intended outcome.
6. Make the smallest valid change.
7. Test the affected flow.
8. Update documentation if behavior changes.

Avoid unrelated refactors during focused fixes.

---

## 20. Source-of-Truth Priority

When information conflicts:

```text
1. verified runtime behavior
2. training/evaluation reports
3. current source code
4. README.md
5. AGENTS.md
6. old notes / assumptions
```

If documentation is stale, update it only after verifying code and reports.

---

## 21. Current Status

Verified:

```text
✅ Kaggle training
✅ EfficientNet-B0 export
✅ FLOOD / NORMAL image classification
✅ local Streamlit inference
✅ confidence display
✅ class probability display
✅ evaluation artifacts
```

Not yet verified:

```text
❌ live CCTV
❌ video temporal reasoning
❌ flood-depth estimation
❌ severity estimation
❌ production alerts
❌ production dashboard
❌ real-world CCTV benchmark
```
