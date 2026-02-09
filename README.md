# Moments - AI English Learning Companion
# Moments - AI English Learning Companion
---
title: Moments - AI English Learning Companion
entry_file: app.py
app_port: 7860
tags:
  - AI
  - English-Learning
  - Multimodal
description: 通过图像与文本交互的英语学习助手，支持外部访问 0.0.0.0:7860，并使用环境变量管理密钥（MODELSCOPE_KEY）。
---

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python app.py
   ```

3. Open your browser at `http://localhost:7860` (or the specific URL provided by your environment).

## Features
- **Capture**: Upload an image to capture a moment.
- **Analyze**: AI identifies objects, actions, and atmosphere (Nouns, Verbs, Adjectives).
- **Compose**: Use the AI-generated tags to write your diary entry.
- **Polish**: AI acts as a teacher to correct and improve your writing.
- **Diary**: View your moments with an encouraging letter from your AI pet.

## Tech Stack
- Python (Flask)
- HTML/CSS (Jinja2)
- ModelScope API (Qwen-VL)
