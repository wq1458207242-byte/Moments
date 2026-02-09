import json
import os
import threading
from app.config import config

class DataService:
    def __init__(self):
        self._locks = {}
        self._global_lock = threading.Lock()

    def _get_lock(self, path):
        with self._global_lock:
            if path not in self._locks:
                self._locks[path] = threading.Lock()
            return self._locks[path]

    def _load_json(self, path, default=None):
        lock = self._get_lock(path)
        with lock:
            try:
                if not os.path.exists(path):
                    return default if default is not None else {}
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data
            except Exception:
                return default if default is not None else {}

    def _save_json(self, path, data):
        lock = self._get_lock(path)
        with lock:
            tmp_path = path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)

    # Word Cards
    def load_word_cards(self):
        data = self._load_json(config.WORD_CARDS_STORE_PATH, default={})
        if isinstance(data, dict):
            return data
        return {}

    def save_word_cards(self, store):
        if isinstance(store, dict):
            self._save_json(config.WORD_CARDS_STORE_PATH, store)

    # Profile
    def load_profile(self):
        default_profile = {
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
        data = self._load_json(config.PROFILE_STORE_PATH, default=default_profile)
        if isinstance(data, dict):
            return data
        return default_profile

    def save_profile(self, profile):
        if isinstance(profile, dict):
            self._save_json(config.PROFILE_STORE_PATH, profile)

    def normalize_profile_numbers(self, profile: dict) -> dict:
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

    def migrate_profile_store_once(self):
        try:
            prof = self.load_profile()
            norm = self.normalize_profile_numbers(prof)
            self.save_profile(norm)
        except Exception:
            pass

    # Users
    def load_users(self):
        data = self._load_json(config.USERS_STORE_PATH, default=[])
        if isinstance(data, list):
            return [u for u in data if isinstance(u, dict)]
        return []

    def save_users(self, users):
        self._save_json(config.USERS_STORE_PATH, users)

    # Moments
    def load_moments(self):
        data = self._load_json(config.MOMENTS_STORE_PATH, default=[])
        if isinstance(data, list):
            return [m for m in data if isinstance(m, dict)]
        return []

    def save_moments(self, moments):
        self._save_json(config.MOMENTS_STORE_PATH, moments)

    # Energy Log
    def append_energy(self, user_id, action, delta):
        try:
            from datetime import datetime
            # Note: This is a complex transaction involving two files.
            # In a DB this would be a transaction. Here we just lock sequentially.
            # Since we are using file-level locks inside _load/_save, we don't have cross-file atomicity here.
            # But at least individual file corruption is prevented.
            
            logs = self._load_json(config.ENERGY_LOG_STORE_PATH, default=[])
            logs.append({
                "user_id": user_id or "anon",
                "action": action,
                "delta": delta,
                "created_at": datetime.utcnow().isoformat() + "Z",
            })
            self._save_json(config.ENERGY_LOG_STORE_PATH, logs)
            
            # Re-load profile to minimize overwrite chance
            profile = self.normalize_profile_numbers(self.load_profile())
            profile["growth_energy"] = int(profile.get("growth_energy", 0)) + int(delta)
            profile["level_text"] = f"Level {int(profile.get('level', 1))}"
            self.save_profile(profile)
        except Exception:
            pass

    def get_energy_today(self):
        energy_today = 0
        try:
            from datetime import date
            logs = self._load_json(config.ENERGY_LOG_STORE_PATH, default=[])
            today_iso = date.today().isoformat()
            for l in logs:
                ts = l.get("created_at","")
                if ts[:10] == today_iso:
                    energy_today += int(l.get("delta",0))
        except Exception:
            energy_today = 0
        return energy_today

    # Letters
    def load_letters(self):
        return self._load_json(config.LETTERS_STORE_PATH, default={})

    def save_letters(self, letters):
        self._save_json(config.LETTERS_STORE_PATH, letters)

    # Diary Store
    def load_diary_store(self):
        return self._load_json(config.DIARY_STORE_PATH, default={})

data_service = DataService()
