from flask import Blueprint, render_template, jsonify, request, url_for
from datetime import datetime, date
import time
from app.services.data_service import data_service
from app.services.ai_service import ai_service
from app.services.cache_service import cache_service
from app.config import config
from app.utils.helpers import _week_strip, _month_grid, _month_abbrev

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    # Calculate streak
    moments = data_service.load_moments()
    active_dates = {m.get("date_key") for m in moments if m.get("date_key")}
    
    streak = 0
    today = date.today()
    current = today
    
    # Check if there is activity today or yesterday to keep streak alive
    if current.isoformat() in active_dates:
        streak += 1
        current -= timedelta(days=1)
    elif (current - timedelta(days=1)).isoformat() in active_dates:
         # If no activity today, but yes yesterday, streak is still valid (but maybe not incremented for today yet? 
         # Usually streak means consecutive days up to now. If I didn't do it today, streak might be X days (from yesterday).
         # Let's count backwards from yesterday.
         current -= timedelta(days=1)
    else:
        # Streak broken
        current = None

    if current:
        while True:
            if current.isoformat() in active_dates:
                streak += 1
                current -= timedelta(days=1)
            else:
                break
                
    return render_template('index.html', streak=streak)

from datetime import timedelta

@main_bp.route('/camera')
def camera():
    return render_template('camera.html')

@main_bp.route('/momentsbook')
def momentsbook():
    moments = data_service.load_moments()
    moments.sort(key=lambda m: (m.get("date_key", ""), m.get("created_at", "")), reverse=True)
    active_dates = {m.get("date_key") for m in moments if m.get("date_key")}
    anchor = date.today()
    if moments and moments[0].get("date_key"):
        try:
            anchor = datetime.strptime(moments[0]["date_key"], "%Y-%m-%d").date()
        except Exception:
            anchor = date.today()

    # Use Today as anchor for calendar view logic to keep it centered on current time
    calendar_anchor = date.today()
    
    week_days = _week_strip(calendar_anchor, active_dates) # Use calendar_anchor instead of last moment
    month_title = _month_abbrev(calendar_anchor)
    
    # counts per date
    date_counts = {}
    for m in moments:
        dk = m.get("date_key") or ""
        if not dk:
            continue
        date_counts[dk] = date_counts.get(dk, 0) + 1
    # inject counts to week strip
    for d in week_days:
        d["count"] = int(date_counts.get(d["date_key"], 0))
        
    months_view = _month_grid(calendar_anchor, months_back=2, months_forward=2, active_date_keys=active_dates, counts=date_counts)

    grouped = {}
    for m in moments:
        dk = m.get("date_key") or ""
        if dk not in grouped:
            grouped[dk] = {
                "date_key": dk,
                "display_date": m.get("display_date") or dk,
                "year": m.get("year") or "",
                "accent": "green" if len(grouped) % 2 == 0 else "dark",
                "moments": [],
            }
        grouped[dk]["moments"].append(m)

    day_sections = list(grouped.values())
    day_sections.sort(key=lambda d: d.get("date_key", ""), reverse=True)
    return render_template(
        "momentsbook.html",
        month_title=month_title,
        week_days=week_days,
        day_sections=day_sections,
        months_view=months_view,
    )

@main_bp.route('/profile', methods=['GET'])
def profile():
    profile = data_service.normalize_profile_numbers(data_service.load_profile())
    
    # Auto-update level logic
    changed = False
    current_energy = int(profile.get("growth_energy", 0))
    current_goal = int(profile.get("growth_goal", 100))
    
    if current_goal < 100:
        current_goal = 100
        changed = True
        
    if current_energy >= current_goal:
        level_int = int(profile.get("level", 1))
        while current_energy >= current_goal:
            current_energy -= current_goal
            level_int += 1
            current_goal = int(current_goal * 1.2)
            changed = True
        
        profile["level"] = level_int
        profile["growth_energy"] = current_energy
        profile["growth_goal"] = current_goal
        
    new_level_text = f"Level {int(profile.get('level', 1))}"
    if profile.get("level_text") != new_level_text:
        profile["level_text"] = new_level_text
        changed = True
        
    if changed:
        data_service.save_profile(profile)

    store = data_service.load_word_cards()
    word_bank_count = len(store)
    return render_template("profile.html", profile=profile, word_bank_count=word_bank_count)

@main_bp.route('/word_bank')
def word_bank():
    store = data_service.load_word_cards()
    items = []
    for key, val in store.items():
        word = val.get("word") or key.split("|")[0]
        items.append({"word": word, "data": val})
    items.sort(key=lambda x: x["word"])
    return render_template("word_bank.html", items=items)

@main_bp.route('/assessment', methods=['GET'])
def assessment_page():
    return render_template('assessment.html')

@main_bp.route('/companion/state')
def companion_state():
    energy_today = data_service.get_energy_today()
    now_ts = time.time()
    
    # Return cached greeting if valid
    if now_ts < cache_service.companion["expiry"] and cache_service.companion["text"]:
        return jsonify({
            "energy_today": energy_today,
            "mood_tag": "supportive",
            "greeting": cache_service.companion["text"]
        })

    # Generate new greeting
    greeting = ai_service.generate_greeting(energy_today)
    
    cache_service.companion["text"] = greeting
    cache_service.companion["expiry"] = now_ts + 600  # Cache for 10 minutes
    
    return jsonify({"energy_today": energy_today, "mood_tag": "supportive", "greeting": greeting})

@main_bp.route('/voice_packs', methods=['GET'])
def voice_packs():
    return jsonify({"items": config.VOICE_PACKS})

@main_bp.route('/profile_voice', methods=['GET'])
def profile_voice():
    profile = data_service.load_profile()
    vp = (profile.get("voice") or {}).get("model") or "piper_zh"
    rate = (profile.get("voice") or {}).get("rate") or 0.95
    pitch = (profile.get("voice") or {}).get("pitch") or config.TTS_PITCH_DEFAULT
    sample = (profile.get("voice") or {}).get("sample") or None
    return jsonify({"model": vp, "rate": rate, "pitch": pitch, "sample": sample})

@main_bp.route('/tts_status', methods=['GET'])
def tts_status():
    return jsonify({"engine": "browser", "model_ready": False, "piper_import_ok": False, "ready": False, "loaded": False, "disabled": True})
