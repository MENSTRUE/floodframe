# 🌊 FloodFrame

FloodFrame is a Computer Vision MVP for early screening of road/camera frames and classifying them into:

- **FLOOD**
- **NORMAL**

The current version is intentionally scoped as an **image/frame classifier**, not yet a full real-time CCTV monitoring system.

---

## 🎯 Problem

Monitoring flood or waterlogging conditions through cameras requires continuous operator attention.

When many camera feeds must be monitored at the same time, important events can be missed or noticed late.

FloodFrame explores the use of Computer Vision to:

- screen road/camera frames automatically,
- identify possible flood or waterlogging indications,
- expose model confidence,
- support faster operator verification.

---

## 👥 Target Users

### Primary
- BPBD
- local government
- area managers
- monitoring operators

### Secondary
- field officers
- public information recipients

---

## 🧠 Current MVP

```text
Image / Camera Frame
        ↓
Preprocessing
        ↓
EfficientNet-B0
        ↓
FLOOD / NORMAL
        ↓
Confidence + Class Probability
        ↓
Operator Verification
```

Input:

```text
.jpg
.jpeg
.png
```

Output:

```text
Prediction
Confidence
Class Probability
```

---

## 🧪 Dataset

### Flood Class — FloodIMG

```text
/kaggle/input/datasets/hhrclemson/flooding-image-dataset
```

Observed:

```text
9,296 flood images
```

### Normal Class — BDD100K

```text
/kaggle/input/datasets/solesensei/solesensei_bdd100k
```

Observed:

```text
136,000 road images
```

Balanced baseline:

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

---

## ⚠️ Dataset Limitation

The FLOOD and NORMAL classes come from different source datasets.

This creates a known risk of **dataset-source bias**. The model may learn source-specific characteristics in addition to flood cues.

Therefore, the experiment metrics must **not** be interpreted as real-world CCTV performance.

Future validation should include:
- external flood images,
- normal roads outside BDD100K,
- wet roads without flooding,
- puddles and reflections,
- night / low-light scenes,
- representative CCTV frames.

---

## 🏗 Model

```text
EfficientNet-B0
```

Training strategy:

```text
ImageNet pretrained weights
→ train classifier head
→ fine-tune last EfficientNet blocks
```

Main configuration:

```text
Input Size       : 224 × 224
Batch Size       : 64
Head Epochs      : 4
Fine-tune Epochs : 8
Optimizer         : AdamW
```

Training was performed on Kaggle GPU.

---

## 📊 Experimental Results

```text
Accuracy        : ≈ 99.89%
Macro F1        : ≈ 99.89%
Flood Precision : 100.00%
Flood Recall    : ≈ 99.78%
Flood F1        : ≈ 99.89%
```

These metrics describe the current experiment test split only.

They do **not** mean:

```text
99.89% real-world CCTV accuracy
```

---

## 🖼 Local Inference Tests

Flood example:

```text
Prediction : FLOOD
Confidence : 100.00%
```

Normal-road example:

```text
Prediction : NORMAL
Confidence : 68.07%
```

Confidence is model confidence only.

It is **not**:
- flood severity,
- water depth,
- danger level.

---

## 🖥 Streamlit Demo

Run:

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

Flow:

```text
Upload frame
→ Preview image
→ Run inference
→ Show FLOOD / NORMAL
→ Show confidence
→ Show class probabilities
```

---

## 📂 Project Structure

```text
floodframe/
├── assets/
│   ├── confusion_matrix.png
│   ├── prediction_grid.png
│   └── training_curves.png
├── data/
│   └── README.md
├── models/
│   ├── README.md
│   └── floodframe_efficientnet_b0.pth
├── notebooks/
│   └── flood.ipynb
├── reports/
│   ├── dataset_distribution.csv
│   ├── final_summary.json
│   ├── grounding_summary.json
│   ├── test_metrics.json
│   └── training_history.csv
├── src/
│   ├── check_dataset.py
│   ├── evaluate.py
│   ├── predict.py
│   └── train.py
├── app.py
├── AGENTS.md
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## 🚀 Training

Notebook:

```text
notebooks/flood.ipynb
```

Primary datasets:

```text
FloodIMG
BDD100K
```

Exported inference model:

```text
models/floodframe_efficientnet_b0.pth
```

---

## 🔮 Product Direction

```text
v0.1  Image / Frame Classification ✅
v0.2  External Image Validation
v0.3  Video Upload
v0.4  Frame Sampling
v0.5  Temporal Persistence Detection
v0.6  Camera Metadata + Location
v0.7  Operator Alert Policy
v0.8  Monitoring Dashboard
v0.9  Edge / Camera Deployment
v1.0  Integrated Flood Monitoring Support System
```

Target future pipeline:

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

---

## 🧑‍💼 Human-in-the-Loop

Principle:

```text
AI gives indication.
Human gives decision.
```

FloodFrame should support operator verification, not replace it.

---

## 🚧 Current Limitations

FloodFrame v0.1 does not yet provide:

- live CCTV stream processing,
- video temporal reasoning,
- flood severity estimation,
- water depth estimation,
- automatic geolocation,
- automatic emergency dispatch,
- production dashboard,
- sensor fusion,
- validated real-world CCTV benchmark.

---

## 📌 Next Step

```text
external image validation
→ video input
→ frame sampling
→ temporal persistence
→ real CCTV validation
```

---

## 📈 Progress

```text
2% — FloodFrame
```

Sedikit demi sedikit, menuju 100%.

---

## 👨‍💻 Author

**Wafa Bila Syaefurokhman**

GitHub: `@MENSTRUE`
