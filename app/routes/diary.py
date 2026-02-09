from flask import Blueprint, request, redirect, url_for, render_template, jsonify
from datetime import datetime, date, timedelta
import uuid
import re
import os
from app.services.data_service import data_service
from app.services.ai_service import ai_service
from app.services.cache_service import cache_service
from app.utils.helpers import _normalize_words, _format_display_date, _month_abbrev
from app.config import config

diary_bp = Blueprint('diary', __name__)

@diary_bp.route('/save_diary', methods=['POST'])
def save_diary():
    data = request.form
    image = (data.get('image') or '').strip()
    content = (data.get('content') or '').strip()
    if not image:
        return redirect(url_for('main.momentsbook'))

    today = date.today()
    date_key = today.isoformat()
    analysis = cache_service.analysis.get(image, {})
    words = _normalize_words(analysis.get("visual_nouns", []))
    scene_hint = str(analysis.get("scene_hint", "")).strip()

    moments = data_service.load_moments()
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
    data_service.save_moments(moments)
    data_service.append_energy("anon", "save_diary", 10)
    return redirect(url_for('main.momentsbook'))

@diary_bp.route('/moments/delete', methods=['POST'])
def moments_delete():
    try:
        data = request.json or {}
        mid = str(data.get("id") or "").strip()
        if not mid:
            return jsonify({"error": "missing id"}), 400
        moments = data_service.load_moments()
        before = len(moments)
        moments = [m for m in moments if str(m.get("id") or "") != mid]
        if len(moments) == before:
            return jsonify({"error": "not_found"}), 404
        data_service.save_moments(moments)
        return jsonify({"ok": True, "deleted_id": mid})
    except Exception as e:
        print(f"Error in moments_delete: {e}")
        return jsonify({"error": str(e)}), 500

@diary_bp.route('/diary/<date_key>')
def diary_by_date(date_key):
    moments = [m for m in data_service.load_moments() if m.get("date_key") == date_key]
    if not moments:
        return redirect(url_for('main.momentsbook'))
    moments.sort(key=lambda m: m.get("created_at", ""))
    first = moments[0]
    image = first.get("image")
    
    # Load existing diary content
    diary_content_str = ""
    # Use load_diary_store from data_service if implemented, or raw load
    # Assuming data_service has it or we do it here.
    # In data_service we added load_diary_store, let's use it or load raw json for now since data_service.load_diary_store returns dict
    d_store = data_service.load_diary_store()
    diary_content_str = d_store.get(date_key, {}).get("content", "")

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

    # Build prompt hint
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
        if objs: parts.append(f"元素：{', '.join(objs)}")
        if acts: parts.append(f"动作：{', '.join(acts)}")
        if moods: parts.append(f"心情：{', '.join(moods)}")
        prompt_hint = "基于今日 " + "；".join(parts) + "，写一段小记吧。"

    # AI Prompts
    moment_texts = [m.get("content", "") for m in moments]
    moment_scenes = [m.get("scene_hint", "") for m in moments]
    combined_context = "; ".join([t for t in moment_texts if t]) + " | " + "; ".join([s for s in moment_scenes if s])
    
    recs = ai_service.generate_diary_prompts(combined_context)
    
    # Brief map
    brief_map = {}
    items = []
    for m in moments[:6]:
        scene = str(m.get("scene_hint",""))
        words = ", ".join((m.get("words") or [])[:5])
        items.append({"id": m.get("id"), "scene": scene, "words": words})
    
    briefs = ai_service.generate_moment_briefs(items)
    for it in briefs:
        i = str(it.get("id","")).strip()
        t = str(it.get("text","")).strip()
        if i and t:
            if len(t) > 96: t = t[:94].rstrip() + "…"
            brief_map[i] = t
            
    # Fallback briefs
    for m in moments:
        if m.get("id") not in brief_map:
            txt = (m.get("content") or "").strip()
            if txt: txt = re.sub(r"\s+", " ", txt)
            if txt:
                brief = txt[:90] + ("…" if len(txt) > 90 else "")
            else:
                brief = "A cozy little moment."
            brief_map[m.get("id")] = brief

    # Letter Logic
    daily_letter = ""
    letters_data = data_service.load_letters()
    if date_key in letters_data:
        target_d = date.fromisoformat(date_key)
        viewable_dt = datetime.combine(target_d + timedelta(days=1), datetime.min.time().replace(hour=7))
        if datetime.now() >= viewable_dt:
            daily_letter = letters_data[date_key].get("content", "")
        else:
            daily_letter = "The spirit is writing... (Available tomorrow at 07:00)"

    data_service.append_energy("anon", "open_daily_view", 1)
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

@diary_bp.route('/diary/letter/generate', methods=['POST'])
def diary_letter_generate():
    date_key = (request.json or {}).get("date_key") or request.form.get("date_key") or ""
    moments = [m for m in data_service.load_moments() if m.get("date_key")==date_key]
    content = "\n".join([m.get("content","") for m in moments])
    letter_text = ai_service.generate_letter(content)
    return render_template("letter.html", letter_text=letter_text)
