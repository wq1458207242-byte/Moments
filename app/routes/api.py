from flask import Blueprint, request, jsonify, redirect, url_for, render_template
import os
import re
import base64
import time
from datetime import datetime
from werkzeug.utils import secure_filename
from app.config import config
from app.services.data_service import data_service
from app.services.ai_service import ai_service
from app.services.cache_service import cache_service
from app.utils.helpers import _pick_visual_nouns, _scene_hint_from_analysis, _compute_tag_positions, _parse_scene_hint, _normalize_words

api_bp = Blueprint('api', __name__)

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

@api_bp.route('/__meta')
def meta():
    from flask import current_app
    index_path = os.path.join(current_app.root_path, 'app', 'templates', 'index.html')
    layout_path = os.path.join(current_app.root_path, 'app', 'templates', 'layout.html')
    return jsonify({
        "root_path": current_app.root_path,
        "debug": current_app.debug,
        "template_auto_reload": bool(getattr(current_app.jinja_env, "auto_reload", False)),
        "index_exists": os.path.exists(index_path),
        "layout_exists": os.path.exists(layout_path),
        "index_mtime": os.path.getmtime(index_path) if os.path.exists(index_path) else None,
        "layout_mtime": os.path.getmtime(layout_path) if os.path.exists(layout_path) else None,
        "routes": sorted([str(r) for r in current_app.url_map.iter_rules()]),
    })

@api_bp.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    image_data_b64 = (request.form.get('image_data') or '').strip()
    filepath = None
    filename = None
    try:
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(config.UPLOAD_FOLDER, filename)
            file.save(filepath)
        elif image_data_b64.startswith('data:image/'):
            try:
                header, b64 = image_data_b64.split(',', 1)
            except ValueError:
                b64 = ''
            if not b64:
                return redirect(url_for('main.camera'))
            raw = base64.b64decode(b64)
            filename = f"capture_{int(datetime.utcnow().timestamp()*1000)}.jpg"
            filepath = os.path.join(config.UPLOAD_FOLDER, filename)
            with open(filepath, 'wb') as f:
                f.write(raw)
        else:
            return redirect(url_for('main.camera'))

        data_service.append_energy("anon", "upload", 5)
        
        profile = data_service.load_profile()
        pref = profile.get("learning_preference", {})
        level = pref.get("target_level", "B1")
        
        analysis = ai_service.analyze_image(filepath, level=level)
        analysis["visual_nouns"] = _pick_visual_nouns(analysis)
        analysis["scene_hint"] = str(analysis.get("scene_hint", ""))
        analysis["tag_positions"] = _compute_tag_positions(analysis)
        cache_service.analysis[filename] = analysis

        next_no = len(data_service.load_moments()) + 1
        moment_code = f"Moment{next_no:03d}"
        return render_template('analysis.html', filename=filename, analysis=analysis, word_cards={}, moment_code=moment_code)
    except Exception as e:
        print(f"Upload error: {e}")
        return redirect(url_for('main.camera'))

@api_bp.route('/upload', methods=['GET'])
def upload_get():
    return redirect(url_for('api.upload_simple'))

@api_bp.route('/upload_simple', methods=['GET'])
def upload_simple():
    return render_template('upload_simple.html')

@api_bp.route('/ai_debug', methods=['GET'])
def ai_debug():
    fn = request.args.get('filename') or ''
    if not fn:
        return jsonify({"error": "missing filename"}), 400
    path = os.path.join(config.UPLOAD_FOLDER, fn)
    if not os.path.exists(path):
        return jsonify({"error": "not_found"}), 404
    out = ai_service.analyze_image(path)
    out["visual_nouns"] = _pick_visual_nouns(out)
    out["scene_hint"] = _scene_hint_from_analysis(out)
    out["tag_positions"] = _compute_tag_positions(out)
    return jsonify(out)

@api_bp.route('/api/word_card')
def api_word_card():
    word = request.args.get('word', '').strip().lower()
    if not word:
        return jsonify({"error": "missing word"}), 400
    scene_hint = request.args.get('scene', '').strip()
    cache_key = f"{word}|{scene_hint}"
    
    if cache_key in cache_service.word_cards:
        cached_mem = cache_service.word_cards.get(cache_key)
        if isinstance(cached_mem, dict) and not cached_mem.get("degraded"):
            return jsonify(cached_mem)
            
    store = data_service.load_word_cards()
    cached = store.get(cache_key)
    if isinstance(cached, dict) and not cached.get("degraded"):
        cache_service.word_cards[cache_key] = cached
        return jsonify(cached)
        
    try:
        card = None
        for _ in range(2):
            try:
                card = ai_service.generate_word_card(word, scene_hint)
                break
            except Exception:
                card = None
        if card is None:
            card = _heuristic_word_card(word, scene_hint)
    except Exception as e:
        print(f"Error generating word card: {e}")
        card = _heuristic_word_card(word, scene_hint)
        
    store[cache_key] = card
    data_service.save_word_cards(store)
    cache_service.word_cards[cache_key] = card
    data_service.append_energy("anon", "open_word_card", 1)
    return jsonify(card)

@api_bp.route('/polish', methods=['POST'])
def polish():
    data = request.json
    text = data.get('text', '')
    result = ai_service.polish_text(text)
    return jsonify(result)

@api_bp.route('/refine', methods=['POST'])
def refine():
    data = request.form
    image = data.get('image')
    content = data.get('content', '')
    content = re.sub(r"\s+", " ", (content or "")).strip()
    refine_data = ai_service.refine_text(content)
    parts, knowledge, recs = build_refine_parts(content, refine_data.get("items", []))
    if not recs:
        parts = [{"type": "text", "content": content}]
        knowledge = {}
    data_service.append_energy("anon", "refine", 5)
    return render_template(
        'refine.html',
        image=image,
        content=content,
        date=datetime.now().strftime("%B %d, %Y"),
        comment=refine_data.get("comment", ""),
        parts=parts,
        knowledge=knowledge,
        recs=recs,
        perfect=refine_data.get("perfect", False),
        score=refine_data.get("score", 0),
        emotional_feedback=refine_data.get("emotional_feedback", ""),
        ghost_words=refine_data.get("ghost_words", []),
    )

@api_bp.route('/refine', methods=['GET'])
def refine_get():
    return redirect(url_for('main.camera'))

@api_bp.route('/word_bank/favorite', methods=['POST'])
def word_bank_favorite():
    data = request.json or {}
    word = (data.get("word") or "").strip().lower()
    scene = (data.get("scene") or "").strip()
    fav = bool(data.get("favorite", True))
    if not word:
        return jsonify({"error": "missing word"}), 400
    key = f"{word}|{scene}"
    store = data_service.load_word_cards()
    if key in store and isinstance(store[key], dict):
        store[key]["favorite"] = fav
        data_service.save_word_cards(store)
        return jsonify({"ok": True, "favorite": fav})
    return jsonify({"error": "not_found"}), 404

@api_bp.route('/assessment/submit', methods=['POST'])
def assessment_submit():
    data = request.json or {}
    raw_score = int(data.get("raw_score", 0))
    ce_fr_level = str(data.get("ce_fr_level", "B1"))
    estimated_vocab = int(data.get("estimated_vocab", 0))
    profile = data_service.load_profile()
    profile["level_text"] = f"Level • {ce_fr_level}"
    lp = profile.get("learning_preference", {})
    lp["target_level"] = ce_fr_level
    profile["learning_preference"] = lp
    profile["estimated_vocab"] = estimated_vocab
    data_service.save_profile(profile)
    data_service.append_energy("anon", "assessment", 5)
    return jsonify({"ok": True, "raw_score": raw_score, "ce_fr_level": ce_fr_level, "estimated_vocab": estimated_vocab})

@api_bp.route('/profile_update', methods=['POST'])
def profile_update():
    profile = data_service.normalize_profile_numbers(data_service.load_profile())
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
    
    if current_energy >= current_goal:
        level_int = int(profile.get("level", 1))
        while current_energy >= current_goal:
            current_energy -= current_goal
            level_int += 1
            current_goal = int(current_goal * 1.2)
            leveled_up = True
            
        profile["level"] = level_int
        profile["growth_energy"] = current_energy
        profile["growth_goal"] = current_goal
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
    data_service.save_profile(profile)
    return jsonify({"ok": True, "profile": profile, "level_up": leveled_up})

@api_bp.route('/profile_avatar_upload', methods=['POST'])
def profile_avatar_upload():
    if 'file' not in request.files:
        return jsonify({"error": "missing file"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "empty filename"}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(config.UPLOAD_FOLDER, filename)
    file.save(filepath)
    profile = data_service.load_profile()
    profile["avatar"] = filename
    data_service.save_profile(profile)
    return jsonify({"ok": True, "avatar": filename, "url": url_for('static', filename='uploads/' + filename, _external=True)})
