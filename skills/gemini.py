import os
import re
import logging
import argparse
import base64
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import traceback
from typing import List, Dict, Any, Tuple, Optional

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from utils.image_utils import smart_new_hw, _decode_base64
from PIL import Image
import io

logger = logging.getLogger(__name__)

def _load_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = "your_gemini_api_key"
    return api_key


class GeminiVLMServer:
    def __init__(self, args) -> None:
        logging.info("Initializing Gemini VLM...")
        
        ##  Token definitions (same as original)
        self.thinking_token_begin = "<thinking>"
        self.thinking_token_end = "</thinking>"
        self.answer_token_begin = "<answer>"
        self.answer_token_end = "</answer>"
        
        ##  Regex patterns (same as original)
        #   System prompt patterns
        self.pattern_subgoal_sentence = re.compile(
            r"^\s*Subgoal:\s*(?P<subgoal>.*?)\s*$",
            re.MULTILINE | re.IGNORECASE
        )
        self.pattern_skill = re.compile(
            r"^\s*Skill:\s*(?P<skill>\d+)\s*$",
            re.MULTILINE | re.IGNORECASE
        )
        self.pattern_view = re.compile(
            r"^\s*View:\s*(?P<view>[^\n\r]+)\s*$",
            re.MULTILINE | re.IGNORECASE
        )
        #   Action prompt patterns
        self.pattern_asset = re.compile(
            r"^\s*Asset:\s*(?P<asset>.*?)\s*$",
            re.MULTILINE | re.IGNORECASE
        )
        self.pattern_point = re.compile(
            r"Point:\s*(\(\s*\d+\s*,\s*\d+\s*\))",
            re.IGNORECASE
        )
        self.pattern_point_3d = re.compile(
            r"Point:\s*(\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\))",
            re.IGNORECASE
        )
        self.pattern_alignment = re.compile(
            r"^\s*Alignment:\s*\(\s*(?P<held_axis>[xyzXYZ])\s*,\s*(?P<target_axis>[xyzXYZ])\s*,\s*(?P<direction>[+-])\s*\)\s*$",
            re.MULTILINE
        )
        self.pattern_rotation = re.compile(
            r"Rotation:\s*(\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\))",
            re.IGNORECASE
        )
        #   Judge prompt patterns
        self.pattern_judge_thinking = re.compile(
            r"^\s*##\s*Thinking\s*$",
            re.MULTILINE | re.IGNORECASE
        )
        self.pattern_judge_summary = re.compile(
            r"^\s*##\s*Summary\s*$",
            re.MULTILINE | re.IGNORECASE
        )
        self.pattern_judge_completion = re.compile(
            r"^\s*##\s*Subgoal Completion(?:\s+and\s+Safety)?\s*$",
            re.MULTILINE | re.IGNORECASE
        )
        self.pattern_judge_reflection = re.compile(
            r"^\s*##\s*Reflection\s*$",
            re.MULTILINE | re.IGNORECASE
        )
        self.pattern_judge_progress = re.compile(
            r"^\s*##\s*Progress Score\s*$",
            re.MULTILINE | re.IGNORECASE
        )
        # self.pattern_judge_progress_score = re.compile(
        #     r"^\s*(?P<score>\d{1,3})\b",
        #     re.MULTILINE
        # )
        self.pattern_judge_progress_score = re.compile(r'\b(\d+)\b')

        if isinstance(args, argparse.Namespace):
            args = vars(args)

        self.model_name = args.get("model", "gemini-2.5-pro")
        self.temperature = args.get("temperature", 1.0)
        self.top_k = args.get("top_k", 10)
        self.top_p = args.get("top_p", 0.95)
        self.max_new_tokens = args.get("max_new_tokens", 4096)

        # Initialize Gemini model
        self.model = self.init_model()


    def init_model(self):
        genai.configure(api_key=_load_api_key())
        
        # Initialize the model
        model_name = self.model_name
        # model_name = "gemini-robotics-er-1.5-preview"

        generation_config = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_new_tokens,
        }
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        return model
    
    
    def base64_to_image_part(self, base64_string: str) -> Dict[str, Any]:
        """Convert base64 string to image part for Gemini API"""
        # Remove data URL prefix if present
        if base64_string.startswith('data:image/'):
            base64_string = base64_string.split(',')[1]
        
        # Decode to get image data
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        # save image for debugging
        return image
    
    
    def _generate_raw(
        self,
        prompt: str = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        max_output_tokens: int = None,
    ) -> str:
        """Generate a single thought using Gemini API"""

        query_text = prompt["text"]
        query_images = prompt.get("images", None) or []

        # Build messages similar to Qwen structure
        contents = []
        user_parts = []

        def _to_image_part(image_data):
            if isinstance(image_data, str):
                return self.base64_to_image_part(image_data)
            return image_data

        text_chunks = query_text.split("<IMG>")
        for idx, chunk in enumerate(text_chunks):
            user_parts.append({"text": chunk})
            if idx < len(text_chunks) - 1 and idx < len(query_images):
                user_parts.append(_to_image_part(query_images[idx]))

        contents.append({
            "role": "user",
            "parts": user_parts
        })

        generation_config = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": int(max_output_tokens) if max_output_tokens else self.max_new_tokens,
        }

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    contents,
                    generation_config=generation_config
                )

                if not response.candidates:
                    raise ValueError("No candidates in response")

                candidate = response.candidates[0]
                # finish_reason: 1=STOP, 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION, 5=OTHER
                finish_reason = candidate.finish_reason
                if finish_reason not in (1, 2):
                    raise ValueError(f"Unexpected finish_reason={finish_reason}")

                if not candidate.content.parts:
                    raise ValueError(f"Empty parts (finish_reason={finish_reason})")

                return candidate.content.parts[0].text

            except Exception as e:
                logging.error(f"Gemini API error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay * (attempt + 1))

        return "Error: max retries exceeded"
    

    def get_point_coords(self, text, h, w):
        def clip_x(x, m, M):
            x = max(x, m)
            x = min(x, M)
            x = int(x)
            return x
        match = self.pattern_point.search(text)
        if match:
            point = list(map(int, eval(match.group(1))))
            h_scale, w_scale = h / 1000.0, w / 1000.0
            point = [point[1] * w_scale, point[0] * h_scale]
            point = [clip_x(point[0], 0, w-1), clip_x(point[1], 0, h-1)]
        else:
            point = None
        return point
    

    def get_skill_id(self, text):
        match = self.pattern_skill.search(text)
        if match:
            skill = int(match.group("skill"))
        else:
            skill = None
        return skill
    

    def get_subgoal_sentence(self, text):
        match = self.pattern_subgoal_sentence.search(text)
        if match:
            subgoal = match.group("subgoal")
        else:
            subgoal = None
        return subgoal
    

    def get_asset_id(self, text):
        # import ipdb; ipdb.set_trace()
        match = self.pattern_asset.search(text)
        if not match:
            return None
        raw = match.group("asset").strip().strip("\"'")
        if raw.isdigit():
            return int(raw)
        return None



    def get_view_id(self, text):
        match = self.pattern_view.search(text)
        if not match:
            return None
        raw = match.group("view").strip().strip("\"'")
        if raw.isdigit():
            return int(raw)
        return None


    def get_point_coords_3d(self, text):
        match = self.pattern_point_3d.search(text)
        if not match:
            return None
        raw = match.group(1).strip().strip("()")
        values = [float(x.strip()) for x in raw.split(",") if x.strip()]
        if len(values) != 3:
            return None
        return values


    def get_alignment(self, text):
        match = self.pattern_alignment.search(text)
        if not match:
            return None
        held_axis = match.group("held_axis").lower()
        target_axis = match.group("target_axis").lower()
        direction = match.group("direction")
        ## Validate values
        if held_axis not in {"x", "y", "z"} or target_axis not in {"x", "y", "z"} or direction not in {"+", "-"}:
            return None
        return (held_axis, target_axis, direction)


    def get_rotation(self, text):
        match = self.pattern_rotation.search(text)
        if not match:
            return None
        raw = match.group(1).strip().strip("()")
        values = [float(x.strip()) for x in raw.split(",") if x.strip()]
        if len(values) != 4:
            return None
        return values


    def get_judge_thinking(self, text):
        match = self.pattern_judge_thinking.search(text)
        if not match:
            return None
        start = match.end()
        next_header = re.search(r"^\s*##\s*", text[start:], re.MULTILINE)
        end = start + next_header.start() if next_header else len(text)
        section = text[start:end].strip()
        return section if section else None


    def get_judge_summary(self, text):
        match = self.pattern_judge_summary.search(text)
        if not match:
            return None
        start = match.end()
        next_header = re.search(r"^\s*##\s*", text[start:], re.MULTILINE)
        end = start + next_header.start() if next_header else len(text)
        section = text[start:end].strip()
        return section if section else None


    def get_judge_completion(self, text):
        match = self.pattern_judge_completion.search(text)
        if not match:
            return None
        start = match.end()
        next_header = re.search(r"^\s*##\s*", text[start:], re.MULTILINE)
        end = start + next_header.start() if next_header else len(text)
        section = text[start:end].strip()
        return section if section else None


    def get_judge_reflection(self, text):
        match = self.pattern_judge_reflection.search(text)
        if not match:
            return None
        start = match.end()
        next_header = re.search(r"^\s*##\s*", text[start:], re.MULTILINE)
        end = start + next_header.start() if next_header else len(text)
        section = text[start:end].strip()
        return section if section else None


    def get_judge_progress(self, text):
        match = self.pattern_judge_progress.search(text)
        if not match:
            return None
        start = match.end()
        next_header = re.search(r"^\s*##\s*", text[start:], re.MULTILINE)
        end = start + next_header.start() if next_header else len(text)
        section = text[start:end].strip()
        return section if section else None


    def get_judge_progress_score(self, progress_text):
        if not progress_text:
            return None
        match = self.pattern_judge_progress_score.search(progress_text)
        if not match:
            return None
        return float(match.group(1))
        

    def parse_text_think_answer(self, text):
        thinking_text = text.split(self.thinking_token_begin)[1].strip()
        # thinking_text = text.split(self.thinking_token_end)[0].strip()
        thinking_text = thinking_text.split(self.thinking_token_end)[0].strip()
        answer_text = text.split(self.answer_token_begin)[1].strip()
        # answer_text = text.split(self.answer_token_end)[0].strip()
        answer_text = answer_text.split(self.answer_token_end)[0].strip()
        return thinking_text, answer_text
    

    def check_format_think_answer(self, text):
        if self.thinking_token_begin not in text:
            return False
        if self.thinking_token_end not in text:
            return False
        if self.answer_token_begin not in text:
            return False
        if self.answer_token_end not in text:
            return False
        return True
    

    def parse_actor_system_results(self, text, **kwargs):
        if not self.check_format_think_answer(text):
            return None, None, None, None
        thinking_text, answer_text = self.parse_text_think_answer(text)
        subgoal_sentence: str = self.get_subgoal_sentence(answer_text)
        skill_id: int = self.get_skill_id(answer_text)
        view_id: int = self.get_view_id(answer_text)
        return thinking_text, skill_id, subgoal_sentence, view_id
    

    def parse_actor_action_results(self, text, **kwargs):
        # import ipdb; ipdb.set_trace()
        if not self.check_format_think_answer(text):
            return None, None, None, None, None, None
        thinking_text, answer_text = self.parse_text_think_answer(text)
        asset_id: int = self.get_asset_id(answer_text)
        point_2d = self.get_point_coords(answer_text, kwargs.get("h", 512), kwargs.get("w", 512))
        alignment = self.get_alignment(answer_text)
        point_3d = self.get_point_coords_3d(answer_text)
        rotation = self.get_rotation(answer_text)
        return thinking_text, asset_id, point_2d, alignment, point_3d, rotation
    

    def parse_judge_subgoal_results(self, text, **kwargs):
        thinking_text = self.get_judge_thinking(text)
        summary_text = self.get_judge_summary(text)
        completion_text = self.get_judge_completion(text)
        return thinking_text, summary_text, completion_text
    

    def parse_judge_progress_results(self, text, **kwargs):
        reflection_text = self.get_judge_reflection(text)
        progress_text = self.get_judge_progress(text)
        progress_score = self.get_judge_progress_score(progress_text)
        return reflection_text, progress_text, progress_score
    

    def generate_single_thought(self, prompt: Dict[str, Any], phase, **kwargs) -> Dict[str, Any]:
        import sys, os
        print(f"[VLM CALL] phase={phase} pid={os.getpid()}", file=sys.stderr, flush=True)
        raw_text = self._generate_raw(prompt)

        if phase == "actor_system":
            thinking_text, skill_id, subgoal_sentence, view_id = self.parse_actor_system_results(raw_text, **kwargs)
            return {
                "thinking_text": thinking_text,
                "skill_id": skill_id,
                "subgoal": subgoal_sentence,
                "view_id": view_id
            }
        elif phase == "actor_action":
            thinking_text, asset_id, point_2d, alignment, point_3d, rotation = self.parse_actor_action_results(raw_text, **kwargs)
            return {
                "thinking_text": thinking_text,
                "asset_id": asset_id,
                "point_2d": point_2d,
                "alignment": alignment,
                "point_3d": point_3d,
                "rotation": rotation,
            }
        elif phase == "judge_subgoal":
            thinking_text, summary_text, completion_text = self.parse_judge_subgoal_results(raw_text, **kwargs)
            return {
                "thinking_text": thinking_text,
                "summary_text": summary_text,
                "completion_text": completion_text
            }
        elif phase == "judge_progress":
            reflection_text, progress_text, progress_score = self.parse_judge_progress_results(raw_text, **kwargs)
            return {
                "reflection_text": reflection_text,
                "progress_text": progress_text,
                "progress_score": progress_score
            }

        else:
            raise ValueError(f"Unknown phase: {phase}")