# 🧠 Email + Calendar RL Environment 📅

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-green.svg)](https://github.com/meta-openenv)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Space-blue)](https://huggingface.co/spaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance, production-ready Reinforcement Learning environment for automated email triage and calendar management. Built for the **Meta OpenEnv Hackathon**.

---

## 🚀 Project Overview

This project simulates a real-world office environment where an AI agent must handle a high volume of incoming emails and complex scheduling requests. The environment is designed to train and evaluate agents on their ability to:
- **Triage**: Identify urgent alerts and spam.
- **Respond**: Provide high-quality, professional replies.
- **Schedule**: Coordinate meetings while avoiding calendar conflicts.

## ✨ Features

- **Hybrid Intelligence**: A sophisticated decision system combining deterministic rule-based logic with LLM reasoning (Qwen 72B). 🧠
- **Persistent Memory**: The agent tracks its actions across steps to ensure no task is repeated or forgotten. 💾
- **Adaptive Reasoning**: Logic shifts dynamically based on current rewards and task progress. 📈
- **OpenEnv Compliant**: Follows strict OpenEnv specifications for observations, actions, and rewards. 📋
- **Multi-Level Difficulty**: Support for `easy`, `medium`, and `hard` task scenarios. ⚖️

---

## 🏗️ Architecture

```mermaid
graph TD
    A[Observation] --> B[Hybrid Agent]
    B --> C{Rule-Based Layer}
    C -- Match Found --> D[Action]
    C -- No Match --> E[LLM Layer]
    E --> D
    D --> F[EmailCalendarEnv]
    F --> G[Grader/Reward]
    G --> A
```

## 📊 Example Performance

The agent achieves **success=true** with optimized efficiency across all difficulty levels.

```text
[START] task=easy env=email-calendar-env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=flag_email reward=0.20 done=false
[STEP] step=4 action=archive_email reward=0.15 done=false
[STEP] step=6 action=reply_email reward=0.20 done=false
[STEP] step=9 action=reply_email reward=0.20 done=true
[END] success=true steps=9 rewards=0.20,0.20,0.20,0.15,0.15,0.20,0.20,0.20,0.20
```

---

## 🛠️ Setup & Installation

### Local Usage
1. **Clone the repo**:
   ```bash
   git clone <repository-url>
   cd email-calendar-env
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set Environment Variables**:
   Create a `.env` file with:
   ```env
   HF_TOKEN=your_huggingface_token
   ```
4. **Run Inference**:
   ```bash
   python inference.py easy
   ```

### Docker Usage
1. **Build**:
   ```bash
   docker build -t email-env .
   ```
2. **Run**:
   ```bash
   docker run -p 7860:7860 email-env
   ```

---

## 🌐 Deployment to Hugging Face Spaces

This project is optimized for Hugging Face Spaces using the Docker SDK.
1. Create a new Space on Hugging Face.
2. Select **Docker** as the SDK.
3. Push the repository to the Space.
4. Add your `HF_TOKEN` as a **Secret** in the Space settings.
5. The Space will automatically build and run the FastAPI server.

## 🏆 Hackathon Submission Checklist

- [x] `inference.py` in root directory
- [x] Strict OpenEnv output format
- [x] Deterministic Grader implementation
- [x] Dockerfile verified for HF Spaces
- [x] Adaptive Hybrid Agent logic

---

**Developed for the Meta OpenEnv Hackathon 2024.** 🚀
