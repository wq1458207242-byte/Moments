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
app.jinja_env.auto_reload = True

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

 

# ModelScope Configuration
_cfg = configparser.ConfigParser()
_cfg.read(os.path.join(app.root_path, "config.ini"), encoding="utf-8")
MODELSCOPE_API_KEY = os.environ.get("MODELSCOPE_API_KEY") or _cfg.get("modelscope", "api_key", fallback="")
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
    api_key=MODELSCOPE_API_KEY,
)

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
                "level": "Level 1 · Beginner",
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
    return {}

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
        profile = _load_profile()
        profile["growth_energy"] = int(profile.get("growth_energy", 0)) + int(delta)
        _save_profile(profile)
    except Exception:
        pass

def _companion_state():
    energy_today = 0
    try:
        logs = []
        if os.path.exists(ENERGY_LOG_STORE_PATH):
            with open(ENERGY_LOG_STORE_PATH, "r", encoding="utf-8") as f:
                logs = json.load(f)
        from datetime import timezone
        today = date.today().isoformat()
        for l in logs:
            ts = l.get("created_at","")
            if ts[:10] == today:
                energy_today += int(l.get("delta",0))
    except Exception:
        energy_today = 0
    mood_tag = "supportive"
    greeting = "Keep going! I’m here with you."
    if energy_today >= 20:
        greeting = "Amazing streak today! Let’s craft another sentence."
    elif energy_today >= 10:
        greeting = "Nice progress! One more moment to reach your goal."
    return {"energy_today": energy_today, "mood_tag": mood_tag, "greeting": greeting}

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

def analyze_image_with_ai(image_path):
    """
    Sends image to ModelScope Qwen-VL to get Nouns, Verbs, and Adjectives.
    """
    try:
        if not MODELSCOPE_API_KEY:
            raise RuntimeError("modelscope_api_key_missing")
        # Encode image to base64
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        prompt = """
        You are an English learning assistant. Analyze this image and return ONLY a valid JSON object.

        Requirements:
        - richness: integer from 1 to 5 (how visually rich / how many distinct elements in the scene)
        - nouns: 6-10 nouns (distinct objects visible)
        - verbs: 4-8 verbs (actions happening or implied)
        - adjectives: 4-8 adjectives (atmosphere or description)
        - boxes: for as many nouns as possible, provide bounding boxes in normalized image coordinates:
          [{"word": "tennis racket", "x": 0.10, "y": 0.20, "w": 0.30, "h": 0.40}]
          where x,y are top-left, w,h are width/height, all in [0,1].

        Return JSON with keys: "richness", "nouns", "verbs", "adjectives", "boxes".
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
        parsed["nouns"] = _normalize_words(parsed.get("nouns", []))
        parsed["verbs"] = _normalize_words(parsed.get("verbs", []))
        parsed["adjectives"] = _normalize_words(parsed.get("adjectives", []))
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
            "nouns": ["moment", "photo", "day"],
            "verbs": ["capturing", "seeing", "feeling"],
            "adjectives": ["nice", "good", "memorable"]
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

        Pick 3 to 5 highlights from the original text. For each highlight:
        - original: an exact substring copied from the user's text (must appear verbatim)
        - improved: a more natural version for that part
        - explanation_cn: a short Chinese explanation of why it's better
        - tone: one of ["native","advanced","casual","formal"]

        Also provide:
        - comment: one encouraging sentence in English for this diary entry.
        Constraints:
        - Tailor suggestions for difficulty={difficulty}, target_level={target_level}
        - Role: {role}

        Return ONLY valid JSON with keys: "comment", "items".
        Example:
        {{"comment":"...","items":[{{"original":"...","improved":"...","explanation_cn":"...","tone":"native"}}]}}
        """

        response = client.chat.completions.create(
            model=TEXT_MODEL_ID,
            messages=[{'role': 'user', 'content': prompt}],
        )
        content = response.choices[0].message.content
        content = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("refine response not a JSON object")
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
        return {"comment": str(data.get("comment", "")).strip() or "Nice moment!", "items": normalized_items[:5]}
    except Exception as e:
        print(f"Error refining text: {e}")
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if not sentences:
            sentences = [text]
        items = []
        for s in sentences[:3]:
            improved = s
            improved = re.sub(r'\bi very like\b', 'I really like', improved, flags=re.IGNORECASE)
            improved = re.sub(r"\bi play tennis\b", "I'm playing tennis", improved, flags=re.IGNORECASE)
            improved = improved[:1].upper() + improved[1:] if improved else improved
            items.append({
                "original": s,
                "improved": improved,
                "explanation_cn": "更自然、更符合英语表达习惯。",
                "tone": "native",
            })
        return {"comment": "Looks like you had a great moment!", "items": items}

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
        analysis = analyze_image_with_ai(filepath)
        analysis["visual_nouns"] = _pick_visual_nouns(analysis)
        analysis["scene_hint"] = _scene_hint_from_analysis(analysis)
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
    - examples_level_en: string
    - examples_level_cn: string
    - examples_scene_en: string
    - examples_scene_cn: string

    Constraints:
    - Keep Chinese concise and natural.
    - Examples should be helpful for an intermediate learner.
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
        recs = []
    _append_energy("anon", "refine", 5)
    return render_template(
        'refine.html',
        image=image,
        content=content,
        date="January 31th, 2026",
        comment=refine_data.get("comment", ""),
        parts=parts,
        knowledge=knowledge,
        recs=recs,
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
    months_view = _month_grid(anchor, months_back=2, months_forward=2, active_date_keys=active_dates, counts=date_counts)

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
    content = "\n".join([str(m.get("content", "")).strip() for m in moments if m.get("content")]).strip()
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
        parsed = _parse_scene_hint(m.get("scene_hint", ""))
        objs.extend(parsed.get("objects", []))
        acts.extend(parsed.get("actions", []))
        moods.extend(parsed.get("mood", []))
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
    recs = []
    base_obj = (objs[0] if objs else "")
    for n in objs[:3]:
        recs.append(f"I really like the {n}.")
    for a in acts[:2]:
        if base_obj:
            recs.append(f"I was {a} near the {base_obj}.")
        else:
            recs.append(f"I was {a} today.")
    for md in moods[:2]:
        recs.append(f"It felt {md}.")
    if not recs:
        recs = ["Today was meaningful.", "I learned something new.", "I want to remember this moment."]
    brief_map = {}
    try:
        items = []
        for m in moments[:6]:
            scene = str(m.get("scene_hint",""))
            words = ", ".join((m.get("words") or [])[:5])
            items.append({"id": m.get("id"), "scene": scene, "words": words})
        prompt = "Create one short, natural English sentence (<= 80 chars) per item, describing the moment casually.\nReturn ONLY JSON list: [{id:'', text:''}] for these items:\n" + json.dumps(items, ensure_ascii=False)
        resp = _chat_completion_with_timeout(model=TEXT_MODEL_ID, messages=[{'role':'user','content':prompt}], timeout_s=18)
        content = resp.choices[0].message.content.strip().replace("```json","").replace("```","")
        data = json.loads(content)
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
    _append_energy("anon", "open_daily_view", 1)
    return render_template(
        "diary.html",
        date=display_date,
        image=image,
        content=content,
        date_key=date_key,
        moments=moments,
        prompt_hint=prompt_hint,
        recs=recs[:6],
        photos_count=len(moments),
        time_hint=time_hint,
        brief_map=brief_map,
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
    profile = _load_profile()
    # collect word bank overview
    store = _load_word_cards_store()
    word_bank_count = len(store)
    return render_template("profile.html", profile=profile, word_bank_count=word_bank_count)

@app.route('/profile_update', methods=['POST'])
def profile_update():
    profile = _load_profile()
    data = request.json or {}
    for k in ["nickname", "level", "role_definition"]:
        if k in data:
            profile[k] = data[k]
    if "growth_energy" in data:
        try:
            profile["growth_energy"] = int(data["growth_energy"])
        except Exception:
            pass
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
    return jsonify({"ok": True, "profile": profile})

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True, use_reloader=False)
