import os
import sys
import json
import base64
import requests
import re
from datetime import datetime, date, timedelta
import uuid
import concurrent.futures
import shutil
import subprocess
import tempfile
import configparser
import threading
import collections
import time
import wave
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from openai import OpenAI

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['JSON_AS_ASCII'] = False
app.jinja_env.auto_reload = True

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

 

# ModelScope Configuration
_cfg = configparser.ConfigParser()
_cfg.read(os.path.join(app.root_path, "config.ini"), encoding="utf-8")
MODELSCOPE_KEY = os.environ.get("MODELSCOPE_KEY") or os.environ.get("MODELSCOPE_API_KEY") or ""
MODELSCOPE_BASE_URL = os.environ.get("MODELSCOPE_BASE_URL") or _cfg.get("modelscope", "base_url", fallback="https://api-inference.modelscope.cn/v1")
def _normalize_model_id(mid):
    try:
        s = str(mid or "").strip()
        s = re.sub(r"\s+", "", s)
        return s
    except Exception:
        return str(mid or "").strip()
MODEL_ID = _normalize_model_id(os.environ.get("MODEL_ID") or _cfg.get("modelscope", "model_id", fallback="Qwen/Qwen2-VL-7B-Instruct"))
MULTIMODAL_MODEL_ID = _normalize_model_id(os.environ.get("MULTIMODAL_MODEL") or _cfg.get("modelscope", "multimodal_model", fallback=None) or MODEL_ID)
TEXT_MODEL_ID = _normalize_model_id(os.environ.get("TEXT_MODEL") or _cfg.get("modelscope", "text_model", fallback=None) or _cfg.get("modelscope", "model_id", fallback="Qwen/Qwen2-7B-Instruct"))
TTS_MAX_PER_MINUTE = int(os.environ.get("TTS_MAX_PER_MINUTE") or _cfg.get("tts", "max_per_minute", fallback="20"))
TTS_MAX_CONCURRENCY = int(os.environ.get("TTS_MAX_CONCURRENCY") or _cfg.get("tts", "max_concurrency", fallback="1"))
TTS_RATE_DEFAULT = float(os.environ.get("TTS_RATE_DEFAULT") or _cfg.get("tts", "rate_default", fallback="1.0"))
TTS_PITCH_DEFAULT = float(os.environ.get("TTS_PITCH_DEFAULT") or _cfg.get("tts", "pitch_default", fallback="1.0"))

client = OpenAI(
    base_url=MODELSCOPE_BASE_URL,
    api_key=MODELSCOPE_KEY,
)
@app.route('/env_status', methods=['GET'])
def env_status():
    return jsonify({
        "MODELSCOPE_KEY_present": bool(MODELSCOPE_KEY),
        "MODELSCOPE_BASE_URL": MODELSCOPE_BASE_URL,
        "MODEL_ID": MODEL_ID
    })

WORD_CARD_CACHE = {}
ANALYSIS_CACHE = {}
MOMENTS_STORE_PATH = os.path.join(app.root_path, "moments_store.json")
WORD_CARDS_STORE_PATH = os.path.join(app.root_path, "word_cards_store.json")
PROFILE_STORE_PATH = os.path.join(app.root_path, "profile_store.json")
USERS_STORE_PATH = os.path.join(app.root_path, "users_store.json")
ENERGY_LOG_STORE_PATH = os.path.join(app.root_path, "energy_log.json")
_TTS_REQ_TIMES = collections.deque()
_TTS_SEM = threading.Semaphore(TTS_MAX_CONCURRENCY)
_PIPER_CACHE = {}
_PIPER_CACHE_LOCK = threading.Lock()

def _load_word_cards_store():
    try:
        if not os.path.exists(WORD_CARDS_STORE_PATH):
            return {}
        with open(WORD_CARDS_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}

def _save_word_cards_store(store):
    if not isinstance(store, dict):
        return
    tmp_path = WORD_CARDS_STORE_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, WORD_CARDS_STORE_PATH)

def _load_profile():
    try:
        if not os.path.exists(PROFILE_STORE_PATH):
            return {
                "nickname": "Alex",
                "avatar": None,
                "level": 1,
                "growth_energy": 20,
                "growth_goal": 100,
                "voice": {
                    "clone_enabled": False,
                    "rate": 0.95,
                    "pitch": 1.0,
                    "sample": None,
                    "model": "piper_zh"
                },
                "role_definition": "Supportive friend",
                "learning_preference": {
                    "difficulty": "medium",
                    "target_level": "B1"
                },
                "notifications": True,
                "privacy": {
                    "share_usage": False
                }
            }
        with open(PROFILE_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {
        "nickname": "Alex",
        "avatar": None,
        "level": 1,
        "growth_energy": 0,
        "growth_goal": 100,
        "voice": {
            "clone_enabled": False,
            "rate": 0.95,
            "pitch": 1.0,
            "sample": None,
            "model": "piper_zh"
        },
        "role_definition": "Supportive friend",
        "learning_preference": {
            "difficulty": "medium",
            "target_level": "B1"
        },
        "notifications": True,
        "privacy": {
            "share_usage": False
        }
    }
def _normalize_profile_numbers(profile: dict) -> dict:
    try:
        raw_level = profile.get("level", 1)
        if isinstance(raw_level, int):
            level_int = raw_level
        else:
            import re
            m = re.search(r"\d+", str(raw_level))
            level_int = int(m.group(0)) if m else 1
        profile["level"] = level_int
    except Exception:
        profile["level"] = 1
    # Normalize energy and goal
    for key, default in [("growth_energy", 0), ("growth_goal", 100)]:
        try:
            val = profile.get(key, default)
            profile[key] = int(val)
        except Exception:
            try:
                import re
                m = re.search(r"\d+", str(val))
                profile[key] = int(m.group(0)) if m else default
            except Exception:
                profile[key] = default
    return profile

def _migrate_profile_store_once():
    try:
        prof = _load_profile()
        norm = _normalize_profile_numbers(prof)
        # Force save normalized profile to avoid legacy string types persisting
        _save_profile(norm)
    except Exception:
        pass

# Run a one-time migration at startup to ensure legacy profiles are normalized
_migrate_profile_store_once()

def _save_profile(profile):
    if not isinstance(profile, dict):
        return
    tmp_path = PROFILE_STORE_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, PROFILE_STORE_PATH)

PIPER_MODEL_DIR = os.path.join(app.root_path, "pretrained_models", "piper")
PIPER_HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
PIPER_VOICES = {
    "piper_zh": {
        "model": "zh_CN-huayan-medium.onnx",
        "config": "zh_CN-huayan-medium.onnx.json",
        "sample_rate": 22050
    },
    "piper_en": {
        "model": "en_US-amy-medium.onnx",
        "config": "en_US-amy-medium.onnx.json",
        "sample_rate": 22050
    }
}
VOICE_PACKS = [
    {"id": "piper_zh", "name": "Piper • zh-CN", "desc": "离线合成（需模型）", "engine": "piper"},
    {"id": "piper_en", "name": "Piper • en-US", "desc": "离线合成（需模型）", "engine": "piper"},
]

def _piper_voice_paths(voice_id):
    if voice_id == "piper_zh":
        return ("zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx", "zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json")
    if voice_id == "piper_en":
        return ("en/en_US/amy/medium/en_US-amy-medium.onnx", "en/en_US/amy/medium/en_US-amy-medium.onnx.json")
    return (None, None)

def _download_file(url, dest):
    import requests
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    r = requests.get(url, stream=True, timeout=60)
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}"
    with open(dest + ".part", "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    os.replace(dest + ".part", dest)
    return True, "ok"

@app.route('/piper/install', methods=['POST'])
def piper_install():
    return jsonify({"error": "feature_disabled", "message": "Server-side TTS disabled"}), 410

@app.route('/piper/download', methods=['POST'])
def piper_download():
    return jsonify({"error": "feature_disabled", "message": "Server-side TTS disabled"}), 410

@app.route('/piper/debug', methods=['GET'])
def piper_debug():
    prof = _load_profile()
    chosen = (prof.get("voice") or {}).get("model") or "piper_zh"
    vcfg = PIPER_VOICES.get(chosen) or {}
    mp = os.path.join(PIPER_MODEL_DIR, vcfg.get("model") or "")
    cp = os.path.join(PIPER_MODEL_DIR, vcfg.get("config") or "")
    return jsonify({"chosen": chosen, "model_path": mp, "config_path": cp, "model_exists": os.path.exists(mp), "config_exists": os.path.exists(cp)})
def _load_users():
    try:
        if not os.path.exists(USERS_STORE_PATH):
            return []
        with open(USERS_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [u for u in data if isinstance(u, dict)]
    except Exception:
        pass
    return []

def _save_users(users):
    tmp_path = USERS_STORE_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, USERS_STORE_PATH)

def _hash(s):
    import hashlib
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def _create_token():
    return uuid.uuid4().hex

@app.route('/auth/register', methods=['POST'])
def auth_register():
    data = request.json or {}
    identifier = (data.get("email") or data.get("phone") or "").strip().lower()
    password = str(data.get("password") or "")
    if not identifier or not password:
        return jsonify({"error": "missing credentials"}), 400
    users = _load_users()
    for u in users:
        if u.get("identifier") == identifier:
            return jsonify({"error": "exists"}), 409
    user_id = uuid.uuid4().hex[:12]
    token = _create_token()
    users.append({
        "id": user_id,
        "identifier": identifier,
        "password_hash": _hash(password),
        "token": token,
        "created_at": datetime.utcnow().isoformat() + "Z",
    })
    _save_users(users)
    return jsonify({"user_id": user_id, "token": token})

@app.route('/auth/login', methods=['POST'])
def auth_login():
    data = request.json or {}
    identifier = (data.get("email") or data.get("phone") or "").strip().lower()
    password = str(data.get("password") or "")
    users = _load_users()
    for u in users:
        if u.get("identifier") == identifier and u.get("password_hash") == _hash(password):
            u["token"] = _create_token()
            _save_users(users)
            return jsonify({"user_id": u.get("id"), "token": u.get("token")})
    return jsonify({"error": "invalid"}), 401

def _append_energy(user_id, action, delta):
    try:
        logs = []
        if os.path.exists(ENERGY_LOG_STORE_PATH):
            with open(ENERGY_LOG_STORE_PATH, "r", encoding="utf-8") as f:
                logs = json.load(f)
        logs.append({
            "user_id": user_id or "anon",
            "action": action,
            "delta": delta,
            "created_at": datetime.utcnow().isoformat() + "Z",
        })
        with open(ENERGY_LOG_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        profile = _normalize_profile_numbers(_load_profile())
        profile["growth_energy"] = int(profile.get("growth_energy", 0)) + int(delta)
        # keep level_text simplified
        profile["level_text"] = f"Level {int(profile.get('level', 1))}"
        _save_profile(profile)
    except Exception:
        pass

COMPANION_CACHE = {
    "text": "",
    "expiry": 0
}

def _companion_state():
    energy_today = 0
    try:
        logs = []
        if os.path.exists(ENERGY_LOG_STORE_PATH):
            with open(ENERGY_LOG_STORE_PATH, "r", encoding="utf-8") as f:
                logs = json.load(f)
        from datetime import timezone
        today_iso = date.today().isoformat()
        for l in logs:
            ts = l.get("created_at","")
            if ts[:10] == today_iso:
                energy_today += int(l.get("delta",0))
    except Exception:
        energy_today = 0

    global COMPANION_CACHE
    now_ts = time.time()
    
    # Return cached greeting if valid
    if now_ts < COMPANION_CACHE["expiry"] and COMPANION_CACHE["text"]:
        return {
            "energy_today": energy_today,
            "mood_tag": "supportive",
            "greeting": COMPANION_CACHE["text"]
        }

    # Generate new greeting via AI
    try:
        profile = _load_profile()
        nickname = profile.get("nickname", "Friend")
        # Ensure we get the CEFR level correctly
        pref = profile.get("learning_preference", {})
        if isinstance(pref, dict):
            level = pref.get("target_level", "B1")
        else:
            level = "B1"
            
        now = datetime.now()
        hour = now.hour
        weekday = now.strftime("%A")
        time_of_day = "morning"
        if 12 <= hour < 18:
            time_of_day = "afternoon"
        elif hour >= 18:
            time_of_day = "evening"
            
        # Get recent moment count
        moments = _load_moments()
        moments_today = [m for m in moments if m.get("date_key") == date.today().isoformat()]
        moment_count = len(moments_today)
        
        prompt = f"""
        You are a supportive language learning companion.
        Context:
        - User: {nickname} (Level {level})
        - Time: {weekday} {time_of_day}
        - Activity today: {moment_count} moments captured, {energy_today} energy points.
        
        Generate a short, natural, warm greeting (max 15 words) that fits this context.
        If they haven't done much (0 moments), encourage them gently.
        If they have done a lot, praise them.
        Do NOT use quotes.
        """
        
        response = _chat_completion_with_timeout(
            model=TEXT_MODEL_ID,
            messages=[{'role': 'user', 'content': prompt}],
            timeout_s=10
        )
        greeting = response.choices[0].message.content.strip()
        # Fallback if empty or error
        if not greeting:
            greeting = "Hello! Ready to capture some moments?"
            
        COMPANION_CACHE["text"] = greeting
        COMPANION_CACHE["expiry"] = now_ts + 600  # Cache for 10 minutes
        
    except Exception as e:
        print(f"Error generating greeting: {e}")
        greeting = "Keep going! I’m here with you."
        
    return {"energy_today": energy_today, "mood_tag": "supportive", "greeting": greeting}

@app.route('/companion/state')
def companion_state():
    return jsonify(_companion_state())
def _parse_scene_hint(scene_hint):
    scene_hint = (scene_hint or "").strip()
    out = {"objects": [], "actions": [], "mood": []}
    if not scene_hint:
        return out
    chunks = [c.strip() for c in scene_hint.split(";") if c.strip()]
    for c in chunks:
        if ":" not in c:
            continue
        k, v = c.split(":", 1)
        k = k.strip().lower()
        items = [x.strip() for x in v.split(",") if x.strip()]
        if "object" in k:
            out["objects"] = items
        elif "action" in k:
            out["actions"] = items
        elif "mood" in k:
            out["mood"] = items
    return out

def _heuristic_word_card(word, scene_hint):
    word = (word or "").strip().lower()
    scene = _parse_scene_hint(scene_hint)
    objs = scene.get("objects", [])
    acts = scene.get("actions", [])
    mood = scene.get("mood", [])
    obj = ""
    for o in objs:
        if o.lower() != word:
            obj = o
            break
    if not obj and objs:
        obj = objs[0]
    act = acts[0] if acts else ""
    mood_word = mood[0] if mood else ""

    pos = "n."
    if word.endswith("ing"):
        pos = "v."
    elif word.endswith("ly"):
        pos = "adv."
    elif word.endswith(("ous", "ful", "ive", "able", "al")):
        pos = "adj."

    level_en = f"I’m learning how to use “{word}” naturally."
    level_cn = f"我在学习如何更自然地使用“{word}”。"
    if mood_word:
        level_en = f"The word “{word}” helps me describe a {mood_word} moment."
        level_cn = f"“{word}”能帮我描述一个更{mood_word}的时刻。"

    scene_en = f"In this photo, “{word}” stands out."
    scene_cn = f"在这张照片里，“{word}”很显眼。"
    if act and obj:
        scene_en = f"In this moment, I’m {act}, and “{word}” connects to the {obj} in the scene."
        scene_cn = f"在这个时刻里，我在{act}，“{word}”与画面里的{obj}呼应。"
    elif obj:
        scene_en = f"In this photo, “{word}” fits well with the {obj}."
        scene_cn = f"在这张照片里，“{word}”和{obj}很搭。"

    return {
        "word": word,
        "phonetic_us": "",
        "phonetic_uk": "",
        "pos": pos,
        "definition_cn": "（AI暂时不可用）先提供一张快速单词卡，稍后可重试获取更详细释义与例句。",
        "examples_level_en": level_en,
        "examples_level_cn": level_cn,
        "examples_scene_en": scene_en,
        "examples_scene_cn": scene_cn,
        "degraded": True,
    }

def _chat_completion_with_timeout(model, messages, timeout_s=20):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(client.chat.completions.create, model=model, messages=messages)
        return fut.result(timeout=timeout_s)

def _load_moments():
    try:
        if not os.path.exists(MOMENTS_STORE_PATH):
            return []
        with open(MOMENTS_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [m for m in data if isinstance(m, dict)]
    except Exception:
        pass
    return []

def _save_moments(moments):
    tmp_path = MOMENTS_STORE_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(moments, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, MOMENTS_STORE_PATH)

def _ordinal(n):
    try:
        n = int(n)
    except Exception:
        return str(n)
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

def _format_display_date(d):
    return f"{d.strftime('%B')} {_ordinal(d.day)}"

def _month_abbrev(d):
    return d.strftime("%b.")  # Jan.

def _week_strip(anchor_date, active_date_keys):
    anchor = anchor_date
    start = anchor - timedelta(days=anchor.weekday())
    days = []
    for i in range(7):
        dd = start + timedelta(days=i)
        days.append({
            "dow": dd.strftime("%a").upper()[:3],
            "day": dd.day,
            "date_key": dd.isoformat(),
            "active": dd.isoformat() in active_date_keys,
            "count": 0,
        })
    return days

def _month_grid(start_month_date, months_back=1, months_forward=0, active_date_keys=None, counts=None):
    active_date_keys = active_date_keys or set()
    counts = counts or {}
    def month_start(d):
        return d.replace(day=1)
    def add_months(d, k):
        y = d.year + (d.month - 1 + k) // 12
        m = (d.month - 1 + k) % 12 + 1
        # clamp day to end of month
        try:
            return date(y, m, 1)
        except Exception:
            return date(d.year, d.month, 1)
    months = []
    start = month_start(start_month_date)
    for k in range(-months_back, months_forward + 1):
        m0 = add_months(start, k)
        # compute weeks grid
        first_weekday = m0.weekday()  # 0 Monday
        # days in month
        if m0.month == 12:
            next_m = date(m0.year + 1, 1, 1)
        else:
            next_m = date(m0.year, m0.month + 1, 1)
        days_count = (next_m - m0).days
        cells = []
        # leading blanks
        for _ in range(first_weekday):
            cells.append(None)
        # month days
        for dday in range(1, days_count + 1):
            d = date(m0.year, m0.month, dday)
            dk = d.isoformat()
            cells.append({
                "date_key": dk,
                "day": dday,
                "dow": d.strftime("%a").upper()[:3],
                "active": dk in active_date_keys,
                "count": int(counts.get(dk, 0)),
            })
        # pad to full weeks (multiples of 7)
        while len(cells) % 7 != 0:
            cells.append(None)
        # split weeks
        weeks = []
        for i in range(0, len(cells), 7):
            weeks.append(cells[i:i+7])
        months.append({
            "month_title": _month_abbrev(m0),
            "year": m0.year,
            "weeks": weeks,
        })
    return months

@app.errorhandler(413)
def request_entity_too_large(e):
    return redirect(url_for('camera'))

def _normalize_words(words):
    if not isinstance(words, list):
        return []
    normalized = []
    seen = set()
    for w in words:
        if not isinstance(w, str):
            continue
        w2 = w.strip()
        if not w2:
            continue
        w2 = w2.lower()
        if w2 in seen:
            continue
        seen.add(w2)
        normalized.append(w2)
    return normalized

def _pick_visual_nouns(analysis):
    nouns = _normalize_words(analysis.get("nouns", []))
    verbs = _normalize_words(analysis.get("verbs", []))
    adjs = _normalize_words(analysis.get("adjectives", []))
    richness = analysis.get("richness")
    try:
        richness = int(richness)
    except Exception:
        richness = None

    if richness is None:
        total = len(nouns) + len(verbs) + len(adjs)
        if total <= 9:
            richness = 2
        elif total <= 15:
            richness = 3
        else:
            richness = 4

    if richness <= 2:
        target = 3
    elif richness == 3:
        target = 4
    else:
        target = 5

    banned = {
        "moment", "photo", "image", "picture", "scene", "day", "thing", "stuff",
        "person", "people", "someone", "somebody",
    }
    filtered = [w for w in nouns if w not in banned]
    if len(filtered) >= target:
        return filtered[:target]
    return (filtered + [w for w in nouns if w not in filtered])[:target]

def _scene_hint_from_analysis(analysis):
    nouns = _normalize_words(analysis.get("nouns", []))[:8]
    verbs = _normalize_words(analysis.get("verbs", []))[:6]
    adjs = _normalize_words(analysis.get("adjectives", []))[:6]
    parts = []
    if nouns:
        parts.append("objects: " + ", ".join(nouns))
    if verbs:
        parts.append("actions: " + ", ".join(verbs))
    if adjs:
        parts.append("mood: " + ", ".join(adjs))
    return "; ".join(parts)

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _extract_boxes(parsed):
    boxes = parsed.get("boxes", [])
    if not isinstance(boxes, list):
        return {}
    out = {}
    for b in boxes:
        if not isinstance(b, dict):
            continue
        word = str(b.get("word", "")).strip().lower()
        if not word:
            continue
        try:
            x = float(b.get("x"))
            y = float(b.get("y"))
            w = float(b.get("w"))
            h = float(b.get("h"))
        except Exception:
            continue
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            continue
        out[word] = {"x": x, "y": y, "w": w, "h": h}
    return out

def _compute_tag_positions(analysis):
    words = _normalize_words(analysis.get("visual_nouns", []))
    boxes = _extract_boxes(analysis)
    if not words:
        return {}

    anchors = []
    golden_angle = 2.399963229728653
    for i, w in enumerate(words):
        b = boxes.get(w)
        if b:
            cx = b["x"] + b["w"] / 2
            cy = b["y"] + b["h"] / 2
        else:
            r = 0.22 + 0.08 * (i % 3)
            a = i * golden_angle
            cx = 0.5 + r * (0.45 * (1 if (i % 2 == 0) else -1)) * (0.6)
            cy = 0.5 + r * (0.35 * (1 if (i % 3 == 0) else -1)) * (0.6)
        anchors.append({"word": w, "ax": _clamp(cx, 0.08, 0.92), "ay": _clamp(cy, 0.20, 0.88)})

    nodes = []
    for a in anchors:
        w = a["word"]
        rr = 0.055 + 0.0035 * min(len(w), 12)
        nodes.append({
            "word": w,
            "x": a["ax"],
            "y": a["ay"],
            "ax": a["ax"],
            "ay": a["ay"],
            "r": rr,
        })

    for _ in range(180):
        for i in range(len(nodes)):
            ni = nodes[i]
            fx = 0.0
            fy = 0.0
            for j in range(len(nodes)):
                if i == j:
                    continue
                nj = nodes[j]
                dx = ni["x"] - nj["x"]
                dy = ni["y"] - nj["y"]
                dist2 = dx * dx + dy * dy + 1e-6
                dist = dist2 ** 0.5
                min_dist = ni["r"] + nj["r"] + 0.01
                if dist < min_dist:
                    push = (min_dist - dist) * 0.6
                    fx += (dx / dist) * push
                    fy += (dy / dist) * push
            fx += (ni["ax"] - ni["x"]) * 0.08
            fy += (ni["ay"] - ni["y"]) * 0.08
            ni["x"] = _clamp(ni["x"] + fx, 0.06, 0.94)
            ni["y"] = _clamp(ni["y"] + fy, 0.18, 0.88)

    positions = {}
    for n in nodes:
        positions[n["word"]] = {"left": round(n["x"] * 100, 2), "top": round(n["y"] * 100, 2)}
    return positions

def analyze_image_with_ai(image_path, level="B1"):
    """
    Sends image to ModelScope Qwen-VL to get Nouns, Verbs, and Adjectives.
    """
    try:
        if not MODELSCOPE_KEY:
            raise RuntimeError("modelscope_api_key_missing")
        # Encode image to base64
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        prompt = f"""
        You are an expert linguistic AI assistant.
        Analyze the image and generate vocabulary for a learner at CEFR Level {level}.

        Return ONLY a JSON object. Do NOT include any "Step 1" text or brainstorming outside the JSON.

        JSON Structure:
        {{
          "richness": 1-5,
          "visual_objects": ["object1", "object2", "object3"], 
          "learning_content": [
            "Advanced Word 1 (Synonym)",
            "Advanced Word 2 (Abstract)",
            "Advanced Word 3 (Descriptive)",
            "Idiomatic Phrase 1",
            "Idiomatic Phrase 2",
            "Complex Sentence Pattern 1"
          ],
          "scene_hint": "Short description of the scene",
          "boxes": []
        }}

        Requirements:
        1. "visual_objects": 
           - Pick exactly 3-5 DISTINCT visible objects for tagging.
           - Use simple, direct nouns (e.g. "book", "lamp", "hand").
        
        2. "learning_content":
           - MUST contain EXACTLY 6 strings.
           - The first 3 strings must be single advanced words (C1/C2 level) related to the image.
           - The next 2 strings must be phrases.
           - The last string must be a sentence pattern.
           - If you cannot find C1 words, use B2 words.
        
        3. "boxes":
           - Provide bounding boxes for "visual_objects" if possible.
        """

        messages_variants = [
            [{
                'role': 'user',
                'content': [
                    {'type': 'input_text', 'text': prompt},
                    {'type': 'input_image', 'image_url': f"data:image/jpeg;base64,{encoded_string}"},
                ],
            }],
            [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{encoded_string}"}},
                ],
            }],
            [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': f"data:image/jpeg;base64,{encoded_string}"},
                ],
            }],
            [{
                'role': 'user',
                'content': prompt,
                'images': [f"data:image/jpeg;base64,{encoded_string}"],
            }],
        ]

        last_err = None
        parsed = None
        for msgs in messages_variants:
            try:
                response = _chat_completion_with_timeout(
                    model=MULTIMODAL_MODEL_ID,
                    messages=msgs,
                    timeout_s=20,
                )
                content = response.choices[0].message.content
                content = content.replace("```json", "").replace("```", "").strip()
                if "{" in content and "}" in content:
                    content = content[content.find("{"):content.rfind("}") + 1]
                parsed_try = json.loads(content)
                if isinstance(parsed_try, dict):
                    parsed = parsed_try
                    break
                last_err = ValueError("parsed content not a dict")
            except Exception as e:
                last_err = e
                parsed = None
        if parsed is None:
            raise last_err or RuntimeError("vision_response_invalid")
        
        # Normalize and map new keys to old keys for compatibility
        # Handle visual_objects (for tags) vs core_words legacy
        visual_objs = parsed.get("visual_objects") or parsed.get("core_words") or []
        parsed["core_words"] = _normalize_words(visual_objs)
        
        # Handle learning_content (for chips) vs support_phrases legacy
        learn_content = parsed.get("learning_content") or parsed.get("support_phrases") or []
        parsed["support_phrases"] = learn_content
        
        # Compatibility mapping
        parsed["visual_nouns"] = parsed["core_words"] 
        parsed["nouns"] = parsed["core_words"]
        parsed["verbs"] = []
        parsed["adjectives"] = []
        
        if not isinstance(parsed.get("boxes"), list):
            parsed["boxes"] = []
        return parsed
    except Exception as e:
        try:
            print(f"Error analyzing image: {e}")
        except Exception:
            pass
        # Fallback data if AI fails
        return {
            "richness": 3,
            "core_words": ["moment", "photo", "day"],
            "support_phrases": ["What a nice day!", "I see something cool.", "Capturing the moment."],
            "scene_hint": "A nice moment",
            "nouns": ["moment", "photo", "day"],
            "verbs": [],
            "adjectives": []
        }

def polish_text_with_ai(text):
    """
    Polishes the user's diary entry.
    """
    try:
        prompt = f"""
        You are an English teacher. The user wrote this diary entry: "{text}".
        Please provide:
        1. A 'corrected' version (fix grammar/spelling).
        2. A 'better' version (more native/natural).
        3. A short 'comment' (encouraging feedback).
        
        Return ONLY a valid JSON object with keys: "corrected", "better", "comment".
        """
        
        response = client.chat.completions.create(
            model=TEXT_MODEL_ID,
            messages=[{'role': 'user', 'content': prompt}],
        )
        
        content = response.choices[0].message.content
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        print(f"Error polishing text: {e}")
        return {
            "corrected": text,
            "better": text,
            "comment": "Great job writing! (AI currently unavailable)"
        }

def refine_text_with_ai(text):
    text = (text or "").strip()
    if not text:
        return {"comment": "Write a few sentences and I'll help refine them.", "items": []}
    try:
        profile = _load_profile()
        role = str(profile.get("role_definition", "")).strip()
        pref = profile.get("learning_preference", {}) if isinstance(profile.get("learning_preference", {}), dict) else {}
        difficulty = str(pref.get("difficulty", "medium"))
        target_level = str(pref.get("target_level", "B1"))
        prompt = f"""
        You are an English writing coach. The user wrote:
        {text}

        Tasks:
        1. Score the user's text from 0 to 100 based on CEFR Level {target_level}.
        2. If score >= 90:
           - Set "perfect": true.
           - Provide a "perfect_msg" (e.g. "Amazing! This is perfect.").
           - No need for "items".
        3. If score < 90:
           - Pick 3-5 highlights to improve.
           - STRICTLY match CEFR Level {target_level} for improvements.
           - For each highlight: original, improved, explanation_cn, tone.
        4. Provide "emotional_feedback": A short, empathetic comment on the content (not the grammar). e.g. "That sounds like a lovely afternoon!"
        5. Provide "ghost_words": 3 possible next words/phrases to continue the story.

        Return ONLY valid JSON with keys: "score", "perfect", "perfect_msg", "emotional_feedback", "ghost_words", "items".
        Example item: {{"original":"...","improved":"...","explanation_cn":"...","tone":"native"}}
        """

        response = client.chat.completions.create(
            model=TEXT_MODEL_ID,
            messages=[{'role': 'user', 'content': prompt}],
        )
        content = response.choices[0].message.content
        content = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        
        # Handle perfect score
        if data.get("perfect"):
            return {
                "perfect": True,
                "score": data.get("score"),
                "comment": data.get("perfect_msg", "Perfect!"),
                "emotional_feedback": data.get("emotional_feedback", ""),
                "ghost_words": data.get("ghost_words", []),
                "items": []
            }

        items = data.get("items", [])
        if not isinstance(items, list):
            items = []
        normalized_items = []
        for it in items:
            if not isinstance(it, dict):
                continue
            original = str(it.get("original", "")).strip()
            improved = str(it.get("improved", "")).strip()
            explanation_cn = str(it.get("explanation_cn", "")).strip()
            tone = str(it.get("tone", "")).strip()
            if not original or not improved:
                continue
            normalized_items.append({
                "original": original,
                "improved": improved,
                "explanation_cn": explanation_cn,
                "tone": tone,
            })
            
        return {
            "perfect": False,
            "score": data.get("score", 0),
            "comment": str(data.get("comment", "")).strip() or "Nice moment!",
            "emotional_feedback": data.get("emotional_feedback", ""),
            "ghost_words": data.get("ghost_words", []),
            "items": normalized_items[:5]
        }
    except Exception as e:
        print(f"Error refining text: {e}")
        # ... fallback logic ...
        return {"comment": "Looks like you had a great moment!", "items": []}

def build_refine_parts(text, items):
    text = text or ""
    palette = ["blue", "green", "yellow", "purple", "pink"]
    highlights = []
    used = []
    for it in items:
        original = (it.get("original") or "").strip()
        if not original:
            continue
        idx = text.find(original)
        if idx < 0:
            idx = text.lower().find(original.lower())
        if idx < 0:
            continue
        end = idx + len(original)
        if any(not (end <= s or idx >= e) for s, e in used):
            continue
        used.append((idx, end))
        hid = f"h{len(highlights)+1}"
        color = palette[(len(highlights)) % len(palette)]
        highlights.append({
            "id": hid,
            "start": idx,
            "end": end,
            "color": color,
            "improved": it.get("improved", ""),
            "explanation_cn": it.get("explanation_cn", ""),
            "tone": it.get("tone", ""),
        })
        if len(highlights) >= 5:
            break

    highlights.sort(key=lambda x: x["start"])
    parts = []
    cursor = 0
    knowledge = {}
    recs = []
    for h in highlights:
        if cursor < h["start"]:
            raw_piece = text[cursor:h["start"]]
            if raw_piece:
                parts.append({"type": "text", "content": raw_piece})
        seg_text = text[h["start"]:h["end"]]
        target_id = f"target-{h['id']}"
        parts.append({"type": "highlight", "content": seg_text, "target_id": target_id, "color": h["color"]})
        knowledge[target_id] = {
            "original": seg_text,
            "improved": h.get("improved", ""),
            "explanation_cn": h.get("explanation_cn", ""),
            "tone": h.get("tone", ""),
            "color": h.get("color", ""),
        }
        recs.append({
            "target_id": target_id,
            "color": h["color"],
            "text": h.get("improved", ""),
            "tone": h.get("tone", ""),
        })
        cursor = h["end"]
    if cursor < len(text):
        tail = text[cursor:]
        if tail:
            parts.append({"type": "text", "content": tail})
    return parts, knowledge, recs

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/__meta')
def meta():
    index_path = os.path.join(app.root_path, 'templates', 'index.html')
    layout_path = os.path.join(app.root_path, 'templates', 'layout.html')
    return jsonify({
        "root_path": app.root_path,
        "debug": app.debug,
        "template_auto_reload": bool(getattr(app.jinja_env, "auto_reload", False)),
        "index_exists": os.path.exists(index_path),
        "layout_exists": os.path.exists(layout_path),
        "index_mtime": os.path.getmtime(index_path) if os.path.exists(index_path) else None,
        "layout_mtime": os.path.getmtime(layout_path) if os.path.exists(layout_path) else None,
        "routes": sorted([str(r) for r in app.url_map.iter_rules()]),
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    image_data_b64 = (request.form.get('image_data') or '').strip()
    filepath = None
    filename = None
    try:
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        elif image_data_b64.startswith('data:image/'):
            try:
                header, b64 = image_data_b64.split(',', 1)
            except ValueError:
                b64 = ''
            if not b64:
                return redirect(url_for('camera'))
            import base64
            raw = base64.b64decode(b64)
            filename = f"capture_{int(datetime.utcnow().timestamp()*1000)}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(filepath, 'wb') as f:
                f.write(raw)
        else:
            return redirect(url_for('camera'))

        _append_energy("anon", "upload", 5)
        
        profile = _load_profile()
        pref = profile.get("learning_preference", {})
        level = pref.get("target_level", "B1")
        
        analysis = analyze_image_with_ai(filepath, level=level)
        analysis["visual_nouns"] = _pick_visual_nouns(analysis)
        analysis["scene_hint"] = str(analysis.get("scene_hint", ""))
        analysis["tag_positions"] = _compute_tag_positions(analysis)
        ANALYSIS_CACHE[filename] = analysis

        next_no = len(_load_moments()) + 1
        moment_code = f"Moment{next_no:03d}"
        return render_template('analysis.html', filename=filename, analysis=analysis, word_cards={}, moment_code=moment_code)
    except Exception:
        return redirect(url_for('camera'))

@app.route('/upload', methods=['GET'])
def upload_get():
    return redirect(url_for('upload_simple'))

@app.route('/upload_simple', methods=['GET'])
def upload_simple():
    return render_template('upload_simple.html')

def _generate_word_card_with_ai(word, scene_hint):
    profile = _load_profile()
    role = str(profile.get("role_definition", "")).strip()
    pref = profile.get("learning_preference", {}) if isinstance(profile.get("learning_preference", {}), dict) else {}
    difficulty = str(pref.get("difficulty", "medium"))
    target_level = str(pref.get("target_level", "B1"))
    prompt = f"""
    You are an English learning assistant. Create a word card for the word: "{word}".
    Scene hint (from user's photo): {scene_hint}
    User preference: difficulty={difficulty}, target_level={target_level}
    Role: {role}

    Output ONLY valid JSON with keys:
    - word: string
    - phonetic_us: string
    - phonetic_uk: string
    - pos: string (e.g. "n.", "v.", "adj.")
    - definition_cn: string (short Chinese definition line, may include multiple items separated by "；")
    - examples_level_en: A natural sentence using the word, suitable for CEFR Level {target_level}.
    - examples_level_cn: Chinese translation.
    - examples_scene_en: A sentence using the word that describes the scene ({scene_hint}), suitable for CEFR Level {target_level}.
    - examples_scene_cn: Chinese translation.

    Constraints:
    - Keep Chinese concise and natural.
    - Examples should be helpful for an intermediate learner.
    - Strictly follow CEFR Level {target_level} for sentence complexity.
    """

    response = _chat_completion_with_timeout(
        model=TEXT_MODEL_ID,
        messages=[{'role': 'user', 'content': prompt}],
        timeout_s=20,
    )
    content = response.choices[0].message.content
    content = content.replace("```json", "").replace("```", "").strip()
    if "{" in content and "}" in content:
        content = content[content.find("{"):content.rfind("}") + 1]
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("word card response is not a JSON object")
    data["word"] = str(data.get("word", word)).strip()
    data["phonetic_us"] = str(data.get("phonetic_us", "")).strip()
    data["phonetic_uk"] = str(data.get("phonetic_uk", "")).strip()
    data["pos"] = str(data.get("pos", "")).strip()
    data["definition_cn"] = str(data.get("definition_cn", "")).strip()
    data["examples_level_en"] = str(data.get("examples_level_en", "")).strip()
    data["examples_level_cn"] = str(data.get("examples_level_cn", "")).strip()
    data["examples_scene_en"] = str(data.get("examples_scene_en", "")).strip()
    data["examples_scene_cn"] = str(data.get("examples_scene_cn", "")).strip()
    data["degraded"] = False
    return data

def _generate_phrase_card_with_ai(text, kind="phrase"):
    profile = _load_profile()
    pref = profile.get("learning_preference", {}) if isinstance(profile.get("learning_preference", {}), dict) else {}
    target_level = str(pref.get("target_level", "B1"))
    prompt = f"""
    You are an English learning assistant. Create a concise learning card for the {kind}: "{text}".
    Output ONLY valid JSON with keys:
    - text: original input
    - translation_cn: short, natural Chinese translation (<= 20 chars)
    - usage_tips: array of 2-3 concise tips (each <= 30 chars, Chinese)
    - examples_en: a CEFR {target_level} level example sentence using the {kind}
    - examples_cn: Chinese translation of the example
    Constraints:
    - Keep it short and beginner-friendly.
    - Avoid markdown code fences.
    """
    response = _chat_completion_with_timeout(
        model=TEXT_MODEL_ID,
        messages=[{'role': 'user', 'content': prompt}],
        timeout_s=15,
    )
    content = response.choices[0].message.content
    content = content.replace("```json", "").replace("```", "").strip()
    if "{" in content and "}" in content:
        content = content[content.find("{"):content.rfind("}") + 1]
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("phrase card response is not a JSON object")
    out = {
        "text": str(data.get("text", text)).strip(),
        "translation_cn": str(data.get("translation_cn", "")).strip(),
        "usage_tips": data.get("usage_tips") if isinstance(data.get("usage_tips"), list) else [],
        "examples_en": str(data.get("examples_en", "")).strip(),
        "examples_cn": str(data.get("examples_cn", "")).strip(),
        "degraded": False,
    }
    return out

@app.route('/ai_debug', methods=['GET'])
def ai_debug():
    fn = request.args.get('filename') or ''
    if not fn:
        return jsonify({"error": "missing filename"}), 400
    path = os.path.join(app.config['UPLOAD_FOLDER'], fn)
    if not os.path.exists(path):
        return jsonify({"error": "not_found"}), 404
    out = analyze_image_with_ai(path)
    out["visual_nouns"] = _pick_visual_nouns(out)
    out["scene_hint"] = _scene_hint_from_analysis(out)
    out["tag_positions"] = _compute_tag_positions(out)
    return jsonify(out)

@app.route('/api/word_card')
def api_word_card():
    word = request.args.get('word', '').strip().lower()
    if not word:
        return jsonify({"error": "missing word"}), 400
    scene_hint = request.args.get('scene', '').strip()
    cache_key = f"{word}|{scene_hint}"
    if cache_key in WORD_CARD_CACHE:
        cached_mem = WORD_CARD_CACHE.get(cache_key)
        if isinstance(cached_mem, dict) and not cached_mem.get("degraded"):
            return jsonify(cached_mem)
    store = _load_word_cards_store()
    cached = store.get(cache_key)
    if isinstance(cached, dict) and not cached.get("degraded"):
        WORD_CARD_CACHE[cache_key] = cached
        return jsonify(cached)
    try:
        last_err = None
        card = None
        for _ in range(2):
            try:
                card = _generate_word_card_with_ai(word, scene_hint)
                break
            except Exception as e:
                last_err = e
                card = None
        if card is None:
            card = _heuristic_word_card(word, scene_hint)
    except Exception as e:
        try:
            print(f"Error generating word card: {e}")
        except Exception:
            pass
        card = _heuristic_word_card(word, scene_hint)
    store[cache_key] = card
    _save_word_cards_store(store)
    WORD_CARD_CACHE[cache_key] = card
    _append_energy("anon", "open_word_card", 1)
    return jsonify(card)

@app.route('/api/phrase_card')
def api_phrase_card():
    q = (request.args.get('q') or "").strip()
    kind = (request.args.get('kind') or "phrase").strip().lower()
    if not q:
        return jsonify({"error": "missing q"}), 400
    try:
        if not MODELSCOPE_KEY:
            raise RuntimeError("modelscope_api_key_missing")
        card = _generate_phrase_card_with_ai(q, kind=kind)
    except Exception:
        card = {
            "text": q,
            "translation_cn": "",
            "usage_tips": [],
            "examples_en": q,
            "examples_cn": "",
            "degraded": True,
        }
    _append_energy("anon", "open_phrase_card", 1)
    return jsonify(card)

@app.route('/polish', methods=['POST'])
def polish():
    data = request.json
    text = data.get('text', '')
    result = polish_text_with_ai(text)
    return jsonify(result)

@app.route('/refine', methods=['POST'])
def refine():
    data = request.form
    image = data.get('image')
    content = data.get('content', '')
    content = re.sub(r"\s+", " ", (content or "")).strip()
    refine_data = refine_text_with_ai(content)
    parts, knowledge, recs = build_refine_parts(content, refine_data.get("items", []))
    if not recs:
        parts = [{"type": "text", "content": content}]
        knowledge = {}
        # recs = [] # Keep recs empty if perfect or no items
    _append_energy("anon", "refine", 5)
    return render_template(
        'refine.html',
        image=image,
        content=content,
        date=datetime.now().strftime("%B %d, %Y"), # Fix static date
        comment=refine_data.get("comment", ""),
        parts=parts,
        knowledge=knowledge,
        recs=recs,
        perfect=refine_data.get("perfect", False),
        score=refine_data.get("score", 0),
        emotional_feedback=refine_data.get("emotional_feedback", ""),
        ghost_words=refine_data.get("ghost_words", []),
    )

@app.route('/refine', methods=['GET'])
def refine_get():
    return redirect(url_for('camera'))

@app.route('/save_diary', methods=['POST'])
def save_diary():
    data = request.form
    image = (data.get('image') or '').strip()
    content = (data.get('content') or '').strip()
    if not image:
        return redirect(url_for('index'))

    today = date.today()
    date_key = today.isoformat()
    analysis = ANALYSIS_CACHE.get(image, {})
    words = _normalize_words(analysis.get("visual_nouns", []))
    scene_hint = str(analysis.get("scene_hint", "")).strip()

    moments = _load_moments()
    moment_id = uuid.uuid4().hex[:12]
    moments.append({
        "id": moment_id,
        "date_key": date_key,
        "display_date": _format_display_date(today),
        "year": today.year,
        "month_abbrev": _month_abbrev(today),
        "image": image,
        "content": content,
        "words": words,
        "scene_hint": scene_hint,
        "created_at": datetime.utcnow().isoformat() + "Z",
    })
    _save_moments(moments)
    _append_energy("anon", "save_diary", 10)
    return redirect(url_for('momentsbook'))

@app.route('/momentsbook')
def momentsbook():
    moments = _load_moments()
    moments.sort(key=lambda m: (m.get("date_key", ""), m.get("created_at", "")), reverse=True)
    active_dates = {m.get("date_key") for m in moments if m.get("date_key")}
    anchor = date.today()
    if moments and moments[0].get("date_key"):
        try:
            anchor = datetime.strptime(moments[0]["date_key"], "%Y-%m-%d").date()
        except Exception:
            anchor = date.today()

    week_days = _week_strip(anchor, active_dates)
    month_title = _month_abbrev(anchor)
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
    # build month grids for browsing (previous month + current month)
    # Correct anchor logic: Use Today as the anchor for month grid view, not the latest moment.
    # This ensures the calendar always centers on or includes the current real-world month.
    calendar_anchor = date.today()
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
@app.route('/moments/delete', methods=['POST'])
def moments_delete():
    try:
        data = request.json or {}
        mid = str(data.get("id") or "").strip()
        if not mid:
            return jsonify({"error": "missing id"}), 400
        moments = _load_moments()
        before = len(moments)
        moments = [m for m in moments if str(m.get("id") or "") != mid]
        if len(moments) == before:
            # It's possible the ID format is different or something, let's debug
            print(f"Delete failed: ID {mid} not found. Available: {[m.get('id') for m in moments]}")
            return jsonify({"error": "not_found"}), 404
        _save_moments(moments)
        return jsonify({"ok": True, "deleted_id": mid})
    except Exception as e:
        print(f"Error in moments_delete: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route('/diary/<date_key>')
def diary_by_date(date_key):
    moments = [m for m in _load_moments() if m.get("date_key") == date_key]
    if not moments:
        return redirect(url_for('momentsbook'))
    moments.sort(key=lambda m: m.get("created_at", ""))
    first = moments[0]
    image = first.get("image")
    
    # Load existing diary content
    diary_content_str = ""
    diary_store_path = os.path.join(app.root_path, "diary_store.json")
    if os.path.exists(diary_store_path):
        try:
            with open(diary_store_path, "r", encoding="utf-8") as f:
                d_store = json.load(f)
                diary_content_str = d_store.get(date_key, {}).get("content", "")
        except:
            pass

    try:
        d = datetime.strptime(date_key, "%Y-%m-%d")
        display_date = f"{d.strftime('%A')}, {d.strftime('%b')} {d.day}"
        try:
            last_ts = (moments[-1].get("created_at") or "").replace("Z", "")
            dt_last = None
            try:
                dt_last = datetime.fromisoformat(last_ts)
            except Exception:
                try:
                    dt_last = datetime.strptime(last_ts, "%Y-%m-%dT%H:%M:%S.%f")
                except Exception:
                    dt_last = datetime.strptime(last_ts, "%Y-%m-%dT%H:%M:%S")
            time_hint = dt_last.strftime('%H:%M %p')
        except Exception:
            time_hint = d.strftime('%H:%M %p')
    except Exception:
        display_date = f"{first.get('display_date','')} · {first.get('year','')}"
        time_hint = "14:30 PM"
    # Build prompt hint and recommended phrases based on scene hints
    objs = []
    acts = []
    moods = []
    for m in moments:
        if m.get("visual_nouns"): objs.extend(m["visual_nouns"])
        if m.get("verbs"): acts.extend(m["verbs"])
        if m.get("adjectives"): moods.extend(m["adjectives"])
    def uniq(xs):
        out = []
        seen = set()
        for x in xs:
            x2 = str(x).strip().lower()
            if not x2 or x2 in seen:
                continue
            seen.add(x2)
            out.append(x2)
        return out
    objs = uniq(objs)[:6]
    acts = uniq(acts)[:4]
    moods = uniq(moods)[:4]
    prompt_hint = "试着写下今天的片段与感受"
    if objs or acts or moods:
        parts = []
        if objs:
            parts.append(f"元素：{', '.join(objs)}")
        if acts:
            parts.append(f"动作：{', '.join(acts)}")
        if moods:
            parts.append(f"心情：{', '.join(moods)}")
        prompt_hint = "基于今日 " + "；".join(parts) + "，写一段小记吧。"
    try:
        # AI-generated prompts
        moment_texts = [m.get("content", "") for m in moments]
        moment_scenes = [m.get("scene_hint", "") for m in moments]
        combined_context = "; ".join([t for t in moment_texts if t]) + " | " + "; ".join([s for s in moment_scenes if s])
        
        prompt = f"""
        You are a reflective journaling companion.
        User's day summary: {combined_context[:1000]}
        
        Generate 3 short, thought-provoking questions (in English) to help the user write their diary.
        Questions should be specific to the day's events if possible, or general if not.
        Return ONLY a valid JSON list of strings: ["Question 1?", "Question 2?", "Question 3?"]
        """
        
        response = _chat_completion_with_timeout(
            model=TEXT_MODEL_ID,
            messages=[{'role': 'user', 'content': prompt}],
            timeout_s=10
        )
        content_json_recs = response.choices[0].message.content.strip()
        if "[" in content_json_recs and "]" in content_json_recs:
            content_json_recs = content_json_recs[content_json_recs.find("["):content_json_recs.rfind("]")+1]
        recs = json.loads(content_json_recs)
        if not isinstance(recs, list):
            recs = ["What was the highlight of your day?", "How did you feel?", "What did you learn?"]
    except Exception as e:
        print(f"Error generating diary prompts: {e}")
        recs = ["What was the highlight of your day?", "How did you feel?", "What did you learn?"]
    brief_map = {}
    try:
        items = []
        for m in moments[:6]:
            scene = str(m.get("scene_hint",""))
            words = ", ".join((m.get("words") or [])[:5])
            items.append({"id": m.get("id"), "scene": scene, "words": words})
        prompt = "Create one short, natural English sentence (<= 80 chars) per item, describing the moment casually.\nReturn ONLY JSON list: [{id:'', text:''}] for these items:\n" + json.dumps(items, ensure_ascii=False)
        resp = _chat_completion_with_timeout(model=TEXT_MODEL_ID, messages=[{'role':'user','content':prompt}], timeout_s=18)
        content_json = resp.choices[0].message.content.strip().replace("```json","").replace("```","")
        data = json.loads(content_json)
        if isinstance(data, list):
            for it in data:
                i = str(it.get("id","")).strip()
                t = str(it.get("text","")).strip()
                if i and t:
                    if len(t) > 96:
                        t = t[:94].rstrip() + "…"
                    brief_map[i] = t
    except Exception:
        for m in moments:
            w = (m.get("words") or [])
            base = ", ".join(w[:2])
            txt = (m.get("content") or "").strip()
            if txt:
                txt = re.sub(r"\s+", " ", txt)
            if base:
                brief = f"{base.capitalize()} and a calm vibe."
            elif txt:
                brief = txt[:90] + ("…" if len(txt) > 90 else "")
            else:
                brief = "A cozy little moment."
            brief_map[m.get("id")] = brief
    
    # Letter Logic
    daily_letter = ""
    try:
        letters_path = os.path.join(app.root_path, "letters_store.json")
        if os.path.exists(letters_path):
            with open(letters_path, "r", encoding="utf-8") as f:
                letters_data = json.load(f)
                if date_key in letters_data:
                    # Check 7:00 AM rule
                    target_d = date.fromisoformat(date_key)
                    viewable_dt = datetime.combine(target_d + timedelta(days=1), datetime.min.time().replace(hour=7))
                    
                    if datetime.now() >= viewable_dt:
                        daily_letter = letters_data[date_key].get("content", "")
                    else:
                        daily_letter = "The spirit is writing... (Available tomorrow at 07:00)"
    except Exception as e:
        print(f"Error loading letter: {e}")

    _append_energy("anon", "open_daily_view", 1)
    return render_template(
        "diary.html",
        date=display_date,
        image=image,
        content=diary_content_str,
        date_key=date_key,
        moments=moments,
        prompt_hint=prompt_hint,
        recs=recs[:6],
        photos_count=len(moments),
        time_hint=time_hint,
        brief_map=brief_map,
        daily_letter=daily_letter,
    )
@app.route('/diary/letter/generate', methods=['POST'])
def diary_letter_generate():
    date_key = (request.json or {}).get("date_key") or request.form.get("date_key") or ""
    moments = [m for m in _load_moments() if m.get("date_key")==date_key]
    content = "\n".join([m.get("content","") for m in moments])
    try:
        profile = _load_profile()
        role = str(profile.get("role_definition","")).strip()
        prompt = f"You are a caring companion. Write a short warm letter summarizing the day: {content}\nRole: {role}"
        response = _chat_completion_with_timeout(model=MODEL_ID, messages=[{'role':'user','content':prompt}], timeout_s=20)
        letter_text = response.choices[0].message.content.strip()
    except Exception:
        letter_text = "It was a meaningful day. Keep going and cherish your moments."
    return render_template("letter.html", letter_text=letter_text)

@app.route('/profile', methods=['GET'])
def profile():
    profile = _normalize_profile_numbers(_load_profile())
    
    # Auto-update level logic on page load to fix legacy data
    changed = False
    current_energy = int(profile.get("growth_energy", 0))
    current_goal = int(profile.get("growth_goal", 100)) # Default to 100 for legacy compatibility
    
    # Ensure reasonable goal start
    if current_goal < 100:
        current_goal = 100
        changed = True
        
    if current_energy >= current_goal:
        # Robustly get current level
        level_int = int(profile.get("level", 1))

        while current_energy >= current_goal:
            current_energy -= current_goal
            level_int += 1
            current_goal = int(current_goal * 1.2)
            changed = True
        
        profile["level"] = level_int
        profile["growth_energy"] = current_energy
        profile["growth_goal"] = current_goal
        
    # Simplify level text as requested
    new_level_text = f"Level {int(profile.get('level', 1))}"
    if profile.get("level_text") != new_level_text:
        profile["level_text"] = new_level_text
        changed = True
        
    if changed:
        _save_profile(profile)

    # collect word bank overview
    store = _load_word_cards_store()
    word_bank_count = len(store)
    return render_template("profile.html", profile=profile, word_bank_count=word_bank_count)

@app.route('/profile_update', methods=['POST'])
def profile_update():
    profile = _normalize_profile_numbers(_load_profile())
    data = request.json or {}
    for k in ["nickname", "level", "role_definition"]:
        if k in data:
            profile[k] = data[k]
    if "growth_energy" in data:
        try:
            profile["growth_energy"] = int(data["growth_energy"])
        except Exception:
            pass
            
    # Check for level up logic
    leveled_up = False
    current_energy = int(profile.get("growth_energy", 0))
    current_goal = int(profile.get("growth_goal", 100))
    
    # Simple level up scheme: if energy exceeds goal
    if current_energy >= current_goal:
        # Robustly get current level
        level_int = int(profile.get("level", 1))

        # Calculate how many levels gained (in case of massive energy gain)
        while current_energy >= current_goal:
            current_energy -= current_goal
            level_int += 1
            # Increase goal by 20% each level
            current_goal = int(current_goal * 1.2)
            leveled_up = True
            
        profile["level"] = level_int
        profile["growth_energy"] = current_energy
        profile["growth_goal"] = current_goal
        
        # Simplify level text
        profile["level_text"] = f"Level {int(profile.get('level', 1))}"

    if "learning_preference" in data and isinstance(data["learning_preference"], dict):
        lp = profile.get("learning_preference", {})
        lp.update(data["learning_preference"])
        profile["learning_preference"] = lp
    if "notifications" in data:
        profile["notifications"] = bool(data["notifications"])
    if "privacy" in data and isinstance(data["privacy"], dict):
        pv = profile.get("privacy", {})
        pv.update(data["privacy"])
        profile["privacy"] = pv
    if "voice" in data and isinstance(data["voice"], dict):
        vc = profile.get("voice", {})
        vc.update(data["voice"])
        profile["voice"] = vc
    _save_profile(profile)
    return jsonify({"ok": True, "profile": profile, "level_up": leveled_up})

@app.route('/voice_packs', methods=['GET'])
def voice_packs():
    return jsonify({"items": VOICE_PACKS})

@app.route('/profile_voice', methods=['GET'])
def profile_voice():
    profile = _load_profile()
    vp = (profile.get("voice") or {}).get("model") or "piper_zh"
    rate = (profile.get("voice") or {}).get("rate") or 0.95
    pitch = (profile.get("voice") or {}).get("pitch") or TTS_PITCH_DEFAULT
    sample = (profile.get("voice") or {}).get("sample") or None
    return jsonify({"model": vp, "rate": rate, "pitch": pitch, "sample": sample})

@app.route('/tts_status', methods=['GET'])
def tts_status():
    return jsonify({"engine": "browser", "model_ready": False, "piper_import_ok": False, "ready": False, "loaded": False, "disabled": True})

@app.route('/voice_sample_upload', methods=['POST'])
def voice_sample_upload():
    return jsonify({"error": "feature_disabled", "message": "Sample upload has been removed."}), 410

@app.route('/tts', methods=['POST'])
def tts():
    return jsonify({"error": "feature_disabled", "message": "Server-side TTS disabled"}), 410

@app.route('/profile_avatar_upload', methods=['POST'])
def profile_avatar_upload():
    if 'file' not in request.files:
        return jsonify({"error": "missing file"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "empty filename"}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    profile = _load_profile()
    profile["avatar"] = filename
    _save_profile(profile)
    return jsonify({"ok": True, "avatar": filename, "url": url_for('static', filename='uploads/' + filename)})

@app.route('/word_bank')
def word_bank():
    store = _load_word_cards_store()
    # group by word for simple view
    items = []
    for key, val in store.items():
        word = val.get("word") or key.split("|")[0]
        items.append({"word": word, "data": val})
    items.sort(key=lambda x: x["word"])
    return render_template("word_bank.html", items=items)

@app.route('/word_bank/favorite', methods=['POST'])
def word_bank_favorite():
    data = request.json or {}
    word = (data.get("word") or "").strip().lower()
    scene = (data.get("scene") or "").strip()
    fav = bool(data.get("favorite", True))
    if not word:
        return jsonify({"error": "missing word"}), 400
    key = f"{word}|{scene}"
    store = _load_word_cards_store()
    if key in store and isinstance(store[key], dict):
        store[key]["favorite"] = fav
        _save_word_cards_store(store)
        return jsonify({"ok": True, "favorite": fav})
    return jsonify({"error": "not_found"}), 404

@app.route('/assessment', methods=['GET'])
def assessment_page():
    return render_template('assessment.html')

@app.route('/assessment/submit', methods=['POST'])
def assessment_submit():
    data = request.json or {}
    raw_score = int(data.get("raw_score", 0))
    ce_fr_level = str(data.get("ce_fr_level", "B1"))
    estimated_vocab = int(data.get("estimated_vocab", 0))
    profile = _load_profile()
    profile["level_text"] = f"Level • {ce_fr_level}"
    lp = profile.get("learning_preference", {})
    lp["target_level"] = ce_fr_level
    profile["learning_preference"] = lp
    profile["estimated_vocab"] = estimated_vocab
    _save_profile(profile)
    _append_energy("anon", "assessment", 5)
    return jsonify({"ok": True, "raw_score": raw_score, "ce_fr_level": ce_fr_level, "estimated_vocab": estimated_vocab})

def _generate_daily_letter_if_needed(target_date_iso):
    letters_path = os.path.join(app.root_path, "letters_store.json")
    letters = {}
    if os.path.exists(letters_path):
        try:
            with open(letters_path, "r", encoding="utf-8") as f:
                letters = json.load(f)
        except:
            letters = {}
            
    if target_date_iso in letters:
        return
        
    # Get moments for that date
    moments = _load_moments()
    day_moments = [m for m in moments if m.get("date_key") == target_date_iso]
    if not day_moments:
        return

    # Generate
    try:
        profile = _load_profile()
        role = str(profile.get("role_definition","")).strip()
        nickname = profile.get("nickname", "Friend")
        pref = profile.get("learning_preference", {})
        level = pref.get("target_level", "B1")
        
        content_summary = "\n".join([f"- {m.get('content','')} ({m.get('scene_hint','')})" for m in day_moments])
        
        prompt = f"""
        You are {role if role else 'a supportive companion'}.
        Write a warm, encouraging letter to {nickname} (Level {level}) about their day ({target_date_iso}).
        
        Moments from the day:
        {content_summary}
        
        Requirements:
        - Reflect on their moments.
        - Be supportive and empathetic.
        - Use simple, warm English suitable for Level {level}.
        - Length: around 100-150 words.
        """
        
        response = _chat_completion_with_timeout(
            model=TEXT_MODEL_ID,
            messages=[{'role': 'user', 'content': prompt}],
            timeout_s=25
        )
        letter_content = response.choices[0].message.content.strip()
        
        letters[target_date_iso] = {
            "content": letter_content,
            "created_at": datetime.now().isoformat()
        }
        
        with open(letters_path, "w", encoding="utf-8") as f:
            json.dump(letters, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Error generating daily letter: {e}")

def background_scheduler():
    while True:
        try:
            now = datetime.now()
            # Run at 00:00 - 00:05
            if now.hour == 0 and now.minute < 5:
                yesterday = (now - timedelta(days=1)).date().isoformat()
                _generate_daily_letter_if_needed(yesterday)
            time.sleep(60)
        except Exception as e:
            print(f"Scheduler error: {e}")
            time.sleep(60)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Start scheduler
    threading.Thread(target=background_scheduler, daemon=True).start()
    
    app.run(host='0.0.0.0', port=7860, debug=True, use_reloader=False)
