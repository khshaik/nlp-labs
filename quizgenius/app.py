"""QuizGenius AI Streamlit frontend."""

from __future__ import annotations

import os

import streamlit as st

from quiz_engine import (
    MAX_QUESTIONS,
    MIN_QUESTIONS,
    VALID_DIFFICULTIES,
    generate_quiz,
)


st.set_page_config(page_title="QuizGenius AI", page_icon="🧠", layout="centered")


def get_api_key() -> str | None:
    """Read the API key without requiring a local secrets file."""
    environment_key = os.getenv("GEMINI_KEY") or os.getenv("GEMINI_API_KEY")
    if environment_key:
        return environment_key
    try:
        return st.secrets.get("GEMINI_KEY") or st.secrets.get("GEMINI_API_KEY")
    except (FileNotFoundError, KeyError):
        return None


def initialize_state() -> None:
    defaults = {
        "quiz_data": None,
        "submitted_answers": None,
        "topic": "",
        "difficulty": "Intermediate",
        "question_count": 3,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_quiz() -> None:
    st.session_state.quiz_data = None
    st.session_state.submitted_answers = None


def render_setup() -> None:
    st.title("🧠 QuizGenius AI")
    st.write("Create a custom AI-generated multiple-choice quiz in seconds.")

    with st.form("quiz_setup"):
        topic = st.text_input(
            "Quiz topic",
            value=st.session_state.topic,
            placeholder="e.g. Natural Language Processing",
        )
        left, right = st.columns(2)
        with left:
            difficulty = st.selectbox(
                "Difficulty",
                VALID_DIFFICULTIES,
                index=VALID_DIFFICULTIES.index(st.session_state.difficulty),
            )
        with right:
            question_count = st.number_input(
                "Number of questions",
                min_value=MIN_QUESTIONS,
                max_value=MAX_QUESTIONS,
                value=st.session_state.question_count,
                step=1,
            )
        submitted = st.form_submit_button("✨ Generate quiz", use_container_width=True)

    if not submitted:
        return
    if not topic.strip():
        st.warning("Enter a topic before generating the quiz.")
        return
    api_key = get_api_key()
    if not api_key:
        st.error(
            "GEMINI_KEY is missing. Export it in the shell or add it to "
            ".streamlit/secrets.toml."
        )
        return

    try:
        with st.spinner("Generating your quiz with Gemini..."):
            quiz = generate_quiz(
                topic=topic,
                difficulty=difficulty,
                num_questions=int(question_count),
                api_key=api_key,
            )
    except Exception as exc:
        st.error(f"Quiz generation failed: {exc}")
        return

    st.session_state.topic = topic.strip()
    st.session_state.difficulty = difficulty
    st.session_state.question_count = int(question_count)
    st.session_state.quiz_data = quiz
    st.session_state.submitted_answers = None
    st.rerun()


def render_results(quiz: dict, answers: dict[int, str]) -> None:
    score = sum(
        answers.get(index) == question["correct_option"]
        for index, question in enumerate(quiz["questions"])
    )
    total = len(quiz["questions"])
    st.subheader(f"🏆 Score: {score}/{total}")
    st.progress(score / total)

    for index, question in enumerate(quiz["questions"]):
        selected = answers.get(index)
        correct = question["correct_option"]
        if selected == correct:
            st.success(f"Q{index + 1}: Correct — {correct}")
        else:
            st.error(f"Q{index + 1}: Your answer: {selected}; correct answer: {correct}")
        st.info(question["explanation"])


def render_quiz() -> None:
    quiz = st.session_state.quiz_data
    st.title(f"✨ {quiz['quiz_title']}")
    st.caption(
        f"{st.session_state.difficulty} • {len(quiz['questions'])} questions • "
        f"Topic: {st.session_state.topic}"
    )

    if st.session_state.submitted_answers is None:
        answers: dict[int, str | None] = {}
        with st.form("quiz_answers"):
            for index, question in enumerate(quiz["questions"]):
                st.subheader(f"Q{index + 1}. {question['text']}")
                option_keys = list(question["options"])
                answers[index] = st.radio(
                    "Select an answer",
                    option_keys,
                    index=None,
                    format_func=lambda key, q=question: f"{key}. {q['options'][key]}",
                    key=f"question_{index}",
                )
                st.divider()
            submitted = st.form_submit_button("Submit answers", use_container_width=True)

        if submitted:
            missing = [index + 1 for index, answer in answers.items() if answer is None]
            if missing:
                st.warning("Answer every question before submitting: " + ", ".join(map(str, missing)))
            else:
                st.session_state.submitted_answers = answers
                st.rerun()
    else:
        render_results(quiz, st.session_state.submitted_answers)

    if st.button("🔄 Create another quiz", use_container_width=True):
        reset_quiz()
        st.rerun()


initialize_state()
if st.session_state.quiz_data is None:
    render_setup()
else:
    render_quiz()
