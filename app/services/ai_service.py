import json
import base64
import concurrent.futures
from datetime import datetime, date
from openai import OpenAI
from app.config import config
from app.services.data_service import data_service
from app.utils.helpers import _normalize_words

class AIService:
    def __init__(self):
        self.client = OpenAI(
            base_url=config.MODELSCOPE_BASE_URL,
            api_key=config.MODELSCOPE_API_KEY,
        )
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

    def _chat_completion_with_timeout(self, model, messages, timeout_s=60):
        fut = self.executor.submit(self.client.chat.completions.create, model=model, messages=messages)
        return fut.result(timeout=timeout_s)

    def analyze_image(self, image_path, level="B1"):
        try:
            if not config.MODELSCOPE_API_KEY:
                raise RuntimeError("modelscope_api_key_missing")
            
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
               - IMPORTANT: Adapt the vocabulary STRICTLY to the learner's level: {level}.
               - CASE Level A1-A2: Use simple, high-frequency nouns (e.g. "tree", "car", "cloud").
               - CASE Level B1-B2: Avoid basic A1 words. Use more specific/descriptive nouns (e.g. "branch", "vehicle", "overcast").
               - CASE Level C1-C2: Use precise, sophisticated, or literary nouns (e.g. "foliage", "chassis", "cumulus").
            
            2. "learning_content":
               - MUST contain EXACTLY 6 strings.
               - The first 3 strings must be single advanced words (C1/C2 level) related to the image.
               - The next 2 strings must be phrases.
               - The last string must be a concrete, usable sentence example describing the scene (e.g. "The sunlight gently touches the desk"), NOT a grammatical structure like "Subject + Verb".
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
                    response = self._chat_completion_with_timeout(
                        model=config.MULTIMODAL_MODEL_ID,
                        messages=msgs,
                        timeout_s=60,
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
            
            visual_objs = parsed.get("visual_objects") or parsed.get("core_words") or []
            parsed["core_words"] = _normalize_words(visual_objs)
            
            learn_content = parsed.get("learning_content") or parsed.get("support_phrases") or []
            parsed["support_phrases"] = learn_content
            
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
            return {
                "richness": 3,
                "core_words": ["moment", "photo", "day"],
                "support_phrases": ["What a nice day!", "I see something cool.", "Capturing the moment."],
                "scene_hint": "A nice moment",
                "nouns": ["moment", "photo", "day"],
                "verbs": [],
                "adjectives": []
            }

    def polish_text(self, text):
        try:
            prompt = f"""
            You are an English teacher. The user wrote this diary entry: "{text}".
            Please provide:
            1. A 'corrected' version (fix grammar/spelling).
            2. A 'better' version (more native/natural).
            3. A short 'comment' (encouraging feedback).
            
            Return ONLY a valid JSON object with keys: "corrected", "better", "comment".
            """
            
            response = self._chat_completion_with_timeout(
                model=config.TEXT_MODEL_ID,
                messages=[{'role': 'user', 'content': prompt}],
                timeout_s=60
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

    def refine_text(self, text):
        text = (text or "").strip()
        if not text:
            return {"comment": "Write a few sentences and I'll help refine them.", "items": []}
        try:
            profile = data_service.load_profile()
            pref = profile.get("learning_preference", {}) if isinstance(profile.get("learning_preference", {}), dict) else {}
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

            response = self._chat_completion_with_timeout(
                model=config.TEXT_MODEL_ID,
                messages=[{'role': 'user', 'content': prompt}],
                timeout_s=60
            )
            content = response.choices[0].message.content
            content = content.replace("```json", "").replace("```", "").strip()
            data = json.loads(content)
            
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
            return {"comment": "Looks like you had a great moment!", "items": []}

    def generate_word_card(self, word, scene_hint):
        profile = data_service.load_profile()
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

        response = self._chat_completion_with_timeout(
            model=config.TEXT_MODEL_ID,
            messages=[{'role': 'user', 'content': prompt}],
            timeout_s=60,
        )
        content = response.choices[0].message.content
        content = content.replace("```json", "").replace("```", "").strip()
        if "{" in content and "}" in content:
            content = content[content.find("{"):content.rfind("}") + 1]
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("word card response is not a JSON object")
        
        # Normalize fields
        for field in ["word", "phonetic_us", "phonetic_uk", "pos", "definition_cn", 
                      "examples_level_en", "examples_level_cn", "examples_scene_en", "examples_scene_cn"]:
            data[field] = str(data.get(field, "")).strip()
            
        data["degraded"] = False
        return data

    def generate_greeting(self, energy_today):
        try:
            profile = data_service.load_profile()
            nickname = profile.get("nickname", "Friend")
            pref = profile.get("learning_preference", {})
            level = pref.get("target_level", "B1")
                
            now = datetime.now()
            hour = now.hour
            weekday = now.strftime("%A")
            time_of_day = "morning"
            if 12 <= hour < 18:
                time_of_day = "afternoon"
            elif hour >= 18:
                time_of_day = "evening"
                
            moments = data_service.load_moments()
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
            
            response = self._chat_completion_with_timeout(
                model=config.TEXT_MODEL_ID,
                messages=[{'role': 'user', 'content': prompt}],
                timeout_s=10
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating greeting: {e}")
            return "Keep going! I’m here with you."

    def generate_diary_prompts(self, combined_context):
        try:
            prompt = f"""
            You are a reflective journaling companion.
            User's day summary: {combined_context[:1000]}
            
            Generate 3 short, thought-provoking questions (in English) to help the user write their diary.
            Questions should be specific to the day's events if possible, or general if not.
            Return ONLY a valid JSON list of strings: ["Question 1?", "Question 2?", "Question 3?"]
            """
            
            response = self._chat_completion_with_timeout(
                model=config.TEXT_MODEL_ID,
                messages=[{'role': 'user', 'content': prompt}],
                timeout_s=10
            )
            content = response.choices[0].message.content.strip()
            if "[" in content and "]" in content:
                content = content[content.find("["):content.rfind("]")+1]
            recs = json.loads(content)
            if not isinstance(recs, list):
                return ["What was the highlight of your day?", "How did you feel?", "What did you learn?"]
            return recs
        except Exception:
            return ["What was the highlight of your day?", "How did you feel?", "What did you learn?"]

    def generate_moment_briefs(self, items):
        try:
            prompt = "Create one short, natural English sentence (<= 80 chars) per item, describing the moment casually.\nReturn ONLY JSON list: [{id:'', text:''}] for these items:\n" + json.dumps(items, ensure_ascii=False)
            resp = self._chat_completion_with_timeout(model=config.TEXT_MODEL_ID, messages=[{'role':'user','content':prompt}], timeout_s=18)
            content_json = resp.choices[0].message.content.strip().replace("```json","").replace("```","")
            return json.loads(content_json)
        except Exception:
            return []

    def generate_letter(self, content):
        try:
            profile = data_service.load_profile()
            role = str(profile.get("role_definition","")).strip()
            prompt = f"You are a caring companion. Write a short warm letter summarizing the day: {content}\nRole: {role}"
            response = self._chat_completion_with_timeout(model=config.MODEL_ID, messages=[{'role':'user','content':prompt}], timeout_s=20)
            return response.choices[0].message.content.strip()
        except Exception:
            return "It was a meaningful day. Keep going and cherish your moments."

    def generate_daily_letter_content(self, day_moments, date_iso):
        try:
            profile = data_service.load_profile()
            role = str(profile.get("role_definition","")).strip()
            nickname = profile.get("nickname", "Friend")
            pref = profile.get("learning_preference", {})
            level = pref.get("target_level", "B1")
            
            content_summary = "\n".join([f"- {m.get('content','')} ({m.get('scene_hint','')})" for m in day_moments])
            
            prompt = f"""
            You are {role if role else 'a supportive companion'}.
            Write a warm, encouraging letter to {nickname} (Level {level}) about their day ({date_iso}).
            
            Moments from the day:
            {content_summary}
            
            Requirements:
            - Reflect on their moments.
            - Be supportive and empathetic.
            - Use simple, warm English suitable for Level {level}.
            - Length: around 100-150 words.
            """
            
            response = self._chat_completion_with_timeout(
                model=config.TEXT_MODEL_ID,
                messages=[{'role': 'user', 'content': prompt}],
                timeout_s=60
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating daily letter: {e}")
            return None

ai_service = AIService()
