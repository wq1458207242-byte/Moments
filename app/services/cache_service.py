class CacheService:
    def __init__(self):
        self.word_cards = {}
        self.analysis = {}
        self.companion = {
            "text": "",
            "expiry": 0
        }
        self.piper = {}

cache_service = CacheService()
