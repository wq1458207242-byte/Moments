import os
import configparser

class Config:
    def __init__(self, root_path):
        self.root_path = root_path
        self._cfg = configparser.ConfigParser()
        self._cfg.read(os.path.join(root_path, "config.ini"), encoding="utf-8")

        # Flask Config
        self.UPLOAD_FOLDER = os.path.join(root_path, 'app', 'static', 'uploads')
        self.MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32MB max
        self.SEND_FILE_MAX_AGE_DEFAULT = 0
        self.TEMPLATES_AUTO_RELOAD = True
        self.JSON_AS_ASCII = False

        # ModelScope / AI Config
        self.MODELSCOPE_API_KEY = os.environ.get("MODELSCOPE_API_KEY") or os.environ.get("MODELSCOPE_KEY") or self._cfg.get("modelscope", "api_key", fallback="")
        self.MODELSCOPE_BASE_URL = os.environ.get("MODELSCOPE_BASE_URL") or self._cfg.get("modelscope", "base_url", fallback="https://api-inference.modelscope.cn/v1")
        
        self.MODEL_ID = self._normalize_model_id(os.environ.get("MODEL_ID") or self._cfg.get("modelscope", "model_id", fallback="Qwen/Qwen2-VL-7B-Instruct"))
        self.MULTIMODAL_MODEL_ID = self._normalize_model_id(os.environ.get("MULTIMODAL_MODEL") or self._cfg.get("modelscope", "multimodal_model", fallback=None) or self.MODEL_ID)
        self.TEXT_MODEL_ID = self._normalize_model_id(os.environ.get("TEXT_MODEL") or self._cfg.get("modelscope", "text_model", fallback=None) or self._cfg.get("modelscope", "model_id", fallback="Qwen/Qwen2-7B-Instruct"))

        # TTS Config
        self.TTS_MAX_PER_MINUTE = int(os.environ.get("TTS_MAX_PER_MINUTE") or self._cfg.get("tts", "max_per_minute", fallback="20"))
        self.TTS_MAX_CONCURRENCY = int(os.environ.get("TTS_MAX_CONCURRENCY") or self._cfg.get("tts", "max_concurrency", fallback="1"))
        self.TTS_RATE_DEFAULT = float(os.environ.get("TTS_RATE_DEFAULT") or self._cfg.get("tts", "rate_default", fallback="1.0"))
        self.TTS_PITCH_DEFAULT = float(os.environ.get("TTS_PITCH_DEFAULT") or self._cfg.get("tts", "pitch_default", fallback="1.0"))

        # Data Store Paths
        self.MOMENTS_STORE_PATH = os.path.join(root_path, "moments_store.json")
        self.WORD_CARDS_STORE_PATH = os.path.join(root_path, "word_cards_store.json")
        self.PROFILE_STORE_PATH = os.path.join(root_path, "profile_store.json")
        self.USERS_STORE_PATH = os.path.join(root_path, "users_store.json")
        self.ENERGY_LOG_STORE_PATH = os.path.join(root_path, "energy_log.json")
        self.LETTERS_STORE_PATH = os.path.join(root_path, "letters_store.json")
        self.DIARY_STORE_PATH = os.path.join(root_path, "diary_store.json")
        
        # Piper Config
        self.PIPER_MODEL_DIR = os.path.join(root_path, "pretrained_models", "piper")
        self.PIPER_VOICES = {
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
        self.VOICE_PACKS = [
            {"id": "piper_zh", "name": "Piper • zh-CN", "desc": "离线合成（需模型）", "engine": "piper"},
            {"id": "piper_en", "name": "Piper • en-US", "desc": "离线合成（需模型）", "engine": "piper"},
        ]

    def _normalize_model_id(self, mid):
        try:
            import re
            s = str(mid or "").strip()
            s = re.sub(r"\s+", "", s)
            return s
        except Exception:
            return str(mid or "").strip()

config = Config(os.getcwd())
