"""Gemini-backed quiz generation logic for the Streamlit application."""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

from google import genai
from google.genai import types

# Homebrew Python does not automatically use the macOS system keychain. Using
# truststore keeps TLS verification enabled while honoring locally trusted CAs.
if sys.platform == "darwin":
    import truststore

    truststore.inject_into_ssl()


VALID_DIFFICULTIES = ("Beginner", "Intermediate", "Advanced")
MIN_QUESTIONS = 1
MAX_QUESTIONS = 10
DEFAULT_MODEL = "gemini-2.5-flash"


QUIZ_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["quiz_title", "questions"],
    "properties": {
        "quiz_title": {"type": "string"},
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "id",
                    "text",
                    "options",
                    "correct_option",
                    "explanation",
                ],
                "properties": {
                    "id": {"type": "integer"},
                    "text": {"type": "string"},
                    "options": {
                        "type": "object",
                        "required": ["A", "B", "C", "D"],
                        "properties": {
                            "A": {"type": "string"},
                            "B": {"type": "string"},
                            "C": {"type": "string"},
                            "D": {"type": "string"},
                        },
                    },
                    "correct_option": {
                        "type": "string",
                        "enum": ["A", "B", "C", "D"],
                    },
                    "explanation": {"type": "string"},
                },
            },
        },
    },
}


def _build_prompt(topic: str, difficulty: str, num_questions: int) -> str:
    return (
        "You are QuizGenius, an accurate and engaging quiz master. "
        f"Create exactly {num_questions} multiple-choice questions about "
        f"{topic!r} at {difficulty.lower()} difficulty. Each question must have "
        "four plausible options labelled A, B, C, and D, exactly one correct "
        "option, and a concise educational explanation. Avoid trick questions, "
        "ambiguity, and repeated questions."
    )


def _validate_quiz(quiz: dict[str, Any], num_questions: int) -> dict[str, Any]:
    questions = quiz.get("questions")
    if not isinstance(quiz.get("quiz_title"), str) or not quiz["quiz_title"].strip():
        raise ValueError("The generated quiz has no title.")
    if not isinstance(questions, list) or len(questions) != num_questions:
        raise ValueError(f"Expected {num_questions} generated questions.")

    for index, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            raise ValueError(f"Question {index} is malformed.")
        question["id"] = index
        options = question.get("options")
        if not isinstance(options, dict) or set(options) != set("ABCD"):
            raise ValueError(f"Question {index} must contain options A, B, C and D.")
        if question.get("correct_option") not in options:
            raise ValueError(f"Question {index} has an invalid correct option.")
        for field in ("text", "explanation"):
            if not isinstance(question.get(field), str) or not question[field].strip():
                raise ValueError(f"Question {index} has an invalid {field}.")

    return quiz


def generate_quiz(
    topic: str,
    difficulty: str = "Intermediate",
    num_questions: int = 3,
    api_key: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Generate and validate a multiple-choice quiz using Gemini."""
    topic = topic.strip()
    if not topic:
        raise ValueError("Enter a quiz topic.")
    if difficulty not in VALID_DIFFICULTIES:
        raise ValueError(f"Difficulty must be one of {VALID_DIFFICULTIES}.")
    if not MIN_QUESTIONS <= num_questions <= MAX_QUESTIONS:
        raise ValueError(
            f"Number of questions must be between {MIN_QUESTIONS} and {MAX_QUESTIONS}."
        )

    resolved_key = api_key or os.getenv("GEMINI_KEY") or os.getenv("GEMINI_API_KEY")
    if not resolved_key:
        raise RuntimeError("GEMINI_KEY is not configured.")

    client = genai.Client(api_key=resolved_key)
    response = client.models.generate_content(
        model=model_name or os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
        contents=_build_prompt(topic, difficulty, num_questions),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=QUIZ_SCHEMA,
        ),
    )
    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    raw_text = response.text.strip()
    markdown_match = re.search(r"```(?:json)?\s*({.*})\s*```", raw_text, re.DOTALL)
    json_text = markdown_match.group(1) if markdown_match else raw_text
    json_text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", json_text)

    try:
        quiz = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Gemini returned invalid quiz JSON.") from exc
    if not isinstance(quiz, dict):
        raise ValueError("Gemini returned an invalid quiz structure.")
    return _validate_quiz(quiz, num_questions)
