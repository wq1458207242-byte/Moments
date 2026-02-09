import requests
import json
import sys

BASE = "http://127.0.0.1:7860"

def get_json(path):
    r = requests.get(BASE + path, timeout=10)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"text": r.text}

def post_json(path, data):
    r = requests.post(BASE + path, headers={"Content-Type": "application/json"}, data=json.dumps(data), timeout=20)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"text": r.text}

def main():
    code, meta = get_json("/__meta")
    print("__meta", code, bool(meta.get("routes")))
    code, status = get_json("/tts_status")
    print("tts_status", code, status.get("ready"), status.get("loaded"))
    code, dbg = get_json("/piper/debug")
    print("piper_debug", code, dbg.get("model_exists"), dbg.get("config_exists"))
    code, wc = get_json("/api/word_card?word=test&scene=")
    print("word_card", code, "ok" if code==200 else "skip")
    if status.get("ready"):
        code, tts = post_json("/tts", {"text": "This is a test.", "rate": 1.0, "pitch": 1.0})
        ok = code==200 and isinstance(tts.get("b64",""), str) and len(tts.get("b64",""))>10
        print("tts", code, ok, tts)
    else:
        print("tts", 503, False)

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print("error", str(e))
        sys.exit(1)
