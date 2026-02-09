# Moments - AI English Learning Companion

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
