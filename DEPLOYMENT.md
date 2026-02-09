# Deployment Guide

## Prerequisites
- Python 3.9+
- pip

## Install
```bash
pip install -r requirements.txt
```

## Configuration
Create `config.ini` in the project root:
```ini
[modelscope]
api_key=
base_url=https://api-inference.modelscope.cn/v1
model_id=Qwen/Qwen3-VL-235B-A22B-Instruct

[tts]
max_per_minute=20
max_concurrency=1
rate_default=1.0
pitch_default=1.0
```
Environment variables can override these keys:
`MODELSCOPE_API_KEY`, `MODELSCOPE_BASE_URL`, `MODEL_ID`, `TTS_MAX_PER_MINUTE`, `TTS_MAX_CONCURRENCY`, `TTS_RATE_DEFAULT`, `TTS_PITCH_DEFAULT`.

## Run
```bash
python app.py
```
Open `http://localhost:7860`.

## Piper TTS
1. Install runtime:
```
POST /piper/install
```
2. Download voice:
```
POST /piper/download {"voice":"piper_zh"} or {"voice":"piper_en"}
```
3. Check:
```
GET /tts_status
```
4. Synthesize:
```
POST /tts {"text":"Hello", "rate":1.0, "pitch":1.0}
```
