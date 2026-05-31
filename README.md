
<div align="center">

# 🚀 ScaleForge

### Production-Grade Distributed Training Infrastructure for Large-Scale AI Models

<p>
  <img src="https://img.shields.io/badge/PyTorch-DDP-EE4C2C?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/FSDP-Model_Sharding-8B5CF6?style=for-the-badge" />
  <img src="https://img.shields.io/badge/ZeRO-Optimizer_Sharding-2563EB?style=for-the-badge" />
  <img src="https://img.shields.io/badge/DeepSpeed-ZeRO--2-0EA5E9?style=for-the-badge" />
</p>

<p>
  <img src="https://img.shields.io/badge/MLflow-Experiment_Tracking-0194E2?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Streamlit-Observability-FF4B4B?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/Kubernetes-Jobs-326CE5?style=for-the-badge&logo=kubernetes" />
  <img src="https://img.shields.io/badge/Tests-13_Passing-success?style=for-the-badge" />
</p>

<b>
Streaming Data → DDP → FSDP → ZeRO → DeepSpeed → MLflow → Kubernetes → Observability
</b>

</div>

---

# 📌 Project Overview

ScaleForge is a portfolio-grade ML Systems and Distributed Training platform designed to demonstrate how modern AI training infrastructure is built.

The project includes:

- Distributed training with DDP
- Fully Sharded Data Parallel (FSDP)
- PyTorch ZeRO optimizer sharding
- DeepSpeed ZeRO-2 integration
- Streaming sharded datasets
- Fault-tolerant checkpointing
- MLflow experiment tracking
- Streamlit observability dashboards
- Bottleneck detection
- Straggler detection
- Docker deployment
- Kubernetes job orchestration

---

# 🏗️ Architecture

```text
YAML Config
    ↓
Experiment Runner
    ↓
Trainer
    ↓
Streaming Dataset + Tiny Transformer
    ↓
DDP / FSDP / ZeRO / DeepSpeed
    ↓
Metrics + Checkpoints
    ↓
MLflow + Streamlit
    ↓
Kubernetes Jobs
```

See: docs/architecture.md

---

# ✨ Features

## Training

- Single-process training
- Distributed Data Parallel (DDP)
- Fully Sharded Data Parallel (FSDP)
- PyTorch ZeroRedundancyOptimizer
- DeepSpeed ZeRO-2 engine
- CUDA-aware device management

## Data

- Synthetic sharded dataset generation
- Streaming dataset loader
- Distributed shard assignment

## Reliability

- Checkpoint save/load
- Resume training
- Fault injection
- FSDP checkpoint support
- ZeRO checkpoint support

## Observability

- Loss tracking
- Tokens/sec throughput
- CPU memory tracking
- GPU memory tracking
- Timing breakdown analysis
- Bottleneck detection
- Straggler detection

## Experiment Tracking

- MLflow
- SQLite backend
- Named runs
- Experiment comparison
- Automated experiment runner

## Platform

- Docker
- Kubernetes namespace
- Persistent Volume Claims
- Kubernetes Jobs
- Run / Logs / Status scripts

---

# 📸 Screenshots

## Dashboard

![Dashboard](dashboard.png)

## Kubernetes Training Job

![Kubernetes](kubernetes.png)

## MLflow Overview

![MLflow Overview](mlflow1.png)

## MLflow Metrics

![MLflow Metrics](mlflow2.png)

---

# 📂 Project Structure

```text
scaleforge/
├── configs/
├── configs/experiments/
├── configs/deepspeed/
├── src/scaleforge/
├── tests/
├── metrics/
├── checkpoints/
├── k8s/
├── scripts/
├── docs/
├── dashboard.py
├── train.py
├── local_ddp.py
├── Dockerfile
├── Dockerfile.deepspeed
└── README.md
```

---

# ⚙️ Setup

## Create Environment

```bash
python -m venv .venv
source .venv/Scripts/activate
```

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Generate Data

```bash
python src/scaleforge/generate_data.py --config configs/config.yaml
```

---

# ▶️ Training

## Single Process

```bash
python train.py
```

## DDP

```bash
python local_ddp.py
```

## FSDP

```bash
python local_ddp.py --config configs/experiments/fsdp.yaml
```

## PyTorch ZeRO

```bash
python local_ddp.py --config configs/experiments/zero.yaml
```

## DeepSpeed ZeRO-2

```bash
docker build -f Dockerfile.deepspeed -t scaleforge-deepspeed:latest .
docker run --rm --gpus all scaleforge-deepspeed:latest
```

---

# 📊 Dashboard

Launch:

```bash
streamlit run dashboard.py
```

Tracks:

- Loss
- Step Time
- Tokens/sec
- CPU Memory
- GPU Memory
- Timing Breakdown
- Bottleneck Analysis
- Distributed Training Health

---

# 📈 MLflow

Launch:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Capabilities:

- Run comparison
- Hyperparameter tracking
- Metric tracking
- Strategy comparison (DDP/FSDP/ZeRO/DeepSpeed)

---

# ☸️ Kubernetes

Deploy:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/training-job.yaml
```

Helper scripts:

```bash
bash k8s/run_job.sh
bash k8s/logs.sh
bash k8s/status.sh
```

---

# 🧪 Testing

```bash
pytest -v
```

Current Status:

```text
13 passed
0 failed
```

Coverage includes:

- Config loading
- Dataset loading
- Streaming datasets
- Model forward pass
- Checkpointing
- Metrics logging
- Strategy selection
- Bottleneck detection
- Straggler detection
- MLflow naming

---

# 🛠 Major Build Milestones

1. Config-driven training system
2. Sharded data generation
3. Streaming dataset pipeline
4. Tiny Transformer model
5. DDP training
6. FSDP training
7. PyTorch ZeRO optimizer sharding
8. DeepSpeed ZeRO-2 engine
9. Checkpointing and resume
10. Fault injection
11. MLflow integration
12. Streamlit observability dashboard
13. Straggler detection
14. Kubernetes deployment
15. Automated experiment runner

---

# 📌 Portfolio Highlights

This project demonstrates:

- Distributed AI Training Infrastructure
- ML Systems Engineering
- Model Sharding
- Optimizer Sharding
- Experiment Tracking
- Kubernetes Orchestration
- Production Observability
- Performance Engineering

---

# 🔮 Future Improvements

- Multi-GPU DeepSpeed
- NCCL cluster training
- Distributed checkpointing
- Ray integration
- Cloud GPU deployment
- Model registry integration

---

# 📄 License

MIT License
