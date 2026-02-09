import hashlib
import uuid
from datetime import datetime, date, timedelta
import re

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

def _hash(s):
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def _create_token():
    return uuid.uuid4().hex

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

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
