import threading
import time
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app import create_app
from app.services.ai_service import ai_service
from app.services.data_service import data_service

app = create_app()

def background_scheduler():
    while True:
        try:
            now = datetime.now()
            # Run at 00:00 - 00:05
            if now.hour == 0 and now.minute < 5:
                yesterday = (now - timedelta(days=1)).date().isoformat()
                # Check if letter exists
                letters = data_service.load_letters()
                if yesterday not in letters:
                    # Check if moments exist
                    moments = data_service.load_moments()
                    day_moments = [m for m in moments if m.get("date_key") == yesterday]
                    if day_moments:
                        content = ai_service.generate_daily_letter_content(day_moments, yesterday)
                        if content:
                            letters[yesterday] = {
                                "content": content,
                                "created_at": datetime.now().isoformat()
                            }
                            data_service.save_letters(letters)
            time.sleep(60)
        except Exception as e:
            print(f"Scheduler error: {e}")
            time.sleep(60)

if __name__ == '__main__':
    # Start scheduler
    if not os.environ.get("WERKZEUG_RUN_MAIN") == "true": # Prevent double run with reloader
        threading.Thread(target=background_scheduler, daemon=True).start()
    
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
