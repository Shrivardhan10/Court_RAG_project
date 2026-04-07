from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from prosecution_pipeline import (  # noqa: E402
    defence_arguments,
    judge_arguments,
    prosecution_arguments,
    run_case,
)


st.set_page_config(
    page_title="Court RAG Reasoning Studio",
    page_icon="⚖️",
    layout="wide",
)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Serif:wght@400;600&display=swap');

          :root {
            --ink: #102a43;
            --paper: #f8f5ef;
            --accent: #d9480f;
            --accent-soft: #ffe8cc;
            --slate: #334e68;
            --mint: #d9f99d;
          }

          .stApp {
            background:
              radial-gradient(1200px 500px at -5% -10%, #ffd8a8 0%, transparent 55%),
              radial-gradient(1000px 500px at 110% 0%, #c7f9cc 0%, transparent 52%),
              linear-gradient(180deg, #fffdf8 0%, #f6f0e7 100%);
            color: var(--ink);
          }

          h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: 0.2px;
          }

          p, li, label, .stMarkdown, .stTextArea textarea {
            font-family: 'IBM Plex Serif', serif;
          }

          .hero {
            border: 1px solid #efd8b6;
            background: rgba(255, 248, 237, 0.92);
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: 0 10px 30px rgba(16, 42, 67, 0.08);
            animation: rise-in 0.45s ease-out;
          }

          .card {
            border: 1px solid #e8dbc9;
            background: #fffdf9;
            border-radius: 14px;
            padding: 12px 14px;
            box-shadow: 0 8px 20px rgba(16, 42, 67, 0.06);
            animation: rise-in 0.5s ease-out;
          }

          .tag {
            display: inline-block;
            margin-right: 8px;
            margin-bottom: 8px;
            padding: 5px 10px;
            border-radius: 999px;
            background: var(--accent-soft);
            color: #7a2e0b;
            font-size: 0.82rem;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
          }

          @keyframes rise-in {
            from { transform: translateY(12px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _sample_case_files() -> List[Path]:
    base = PROJECT_ROOT / "processed_cases"
    if not base.exists():
        return []
    return sorted(base.glob("*.json"))


def _trim_text(text: str, limit: int = 4200) -> str:
    cleaned = " ".join((text or "").split())
    return cleaned[:limit].strip()


@st.cache_data(show_spinner=False)
def _load_case_text(file_path: str) -> Dict[str, str]:
    path = Path(file_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    case_name = str(payload.get("case_name") or path.stem)

    facts_parts: List[str] = []
    for chunk in payload.get("chunks", []):
        if str(chunk.get("chunk_type", "")).lower() in {"facts", "issue", "argument"}:
            txt = str(chunk.get("text", "")).strip()
            if txt:
                facts_parts.append(txt)
        if len(facts_parts) >= 2:
            break

    fallback = str(payload.get("summary", "")).strip()
    combined = "\n\n".join(facts_parts).strip() if facts_parts else fallback
    return {
        "case_name": case_name,
        "text": _trim_text(combined),
    }


def _header() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1 style="margin: 0 0 8px 0;">Court RAG Reasoning Studio</h1>
          <p style="margin: 0 0 10px 0; color: #334e68;">
            Practice courtroom reasoning as counsel, compare both sides, and study a reasoned judicial outcome.
          </p>
          <span class="tag">Not a pure outcome predictor</span>
          <span class="tag">Learning-focused legal reasoning</span>
          <span class="tag">Prosecution • Defence • Judge</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_case_input() -> str:
    st.subheader("Case Input")
    input_mode = st.radio(
        "Choose input source",
        ["Manual text", "Load from processed_cases"],
        horizontal=True,
    )

    if "case_text" not in st.session_state:
        st.session_state.case_text = ""

    if input_mode == "Load from processed_cases":
        files = _sample_case_files()
        if not files:
            st.warning("No sample files found in processed_cases. Switch to manual text.")
        else:
            labels = [f.stem for f in files]
            selected_label = st.selectbox("Choose a sample case", labels)
            selected_path = next(f for f in files if f.stem == selected_label)
            sample = _load_case_text(str(selected_path))
            st.caption(f"Loaded: {sample['case_name']}")
            st.text_area("Sample preview", sample["text"], height=180, disabled=True)
            if st.button("Use this sample as case text", use_container_width=True):
                st.session_state.case_text = sample["text"]

    st.session_state.case_text = st.text_area(
        "Case facts / case narrative",
        value=st.session_state.case_text,
        placeholder="Enter the case facts for legal reasoning...",
        height=220,
    )

    return st.session_state.case_text.strip()


def _show_outputs(title: str, text: str) -> None:
    st.markdown(f"<div class='card'><h3 style='margin-top:0'>{title}</h3><p>{text}</p></div>", unsafe_allow_html=True)


def _compare_mode(case_text: str) -> None:
    st.subheader("Mode: Compare Both Sides")
    st.write("Generate prosecution and defence arguments, then view the judge's final order.")

    if st.button("Run Full Comparison", type="primary", use_container_width=True):
        if not case_text:
            st.error("Please provide case text before running.")
            return
        with st.spinner("Running prosecution, defence, and judge reasoning..."):
            try:
                result = run_case(case_text)
            except Exception as exc:
                st.exception(exc)
                return

        col1, col2 = st.columns(2)
        with col1:
            _show_outputs("Prosecution", result.get("prosecution", ""))
        with col2:
            _show_outputs("Defence", result.get("defence", ""))

        _show_outputs("Judge Outcome", result.get("judge", ""))
        st.download_button(
            "Download Result JSON",
            data=json.dumps(result, ensure_ascii=False, indent=2),
            file_name="court_rag_compare_result.json",
            mime="application/json",
            use_container_width=True,
        )


def _role_play_mode(case_text: str) -> None:
    st.subheader("Mode: Role-Play")
    st.write("Pick your side, draft your argument, then let the system generate the opposing side and judge order.")

    role = st.radio("Your role", ["Prosecution", "Defence"], horizontal=True)
    user_key = "user_draft_prosecution" if role == "Prosecution" else "user_draft_defence"

    if st.button("Generate Draft For My Role", use_container_width=True):
        if not case_text:
            st.error("Please provide case text before generating a draft.")
            return
        with st.spinner(f"Generating {role.lower()} draft..."):
            try:
                if role == "Prosecution":
                    st.session_state[user_key] = prosecution_arguments(case_text)
                else:
                    prosecution_seed = prosecution_arguments(case_text)
                    st.session_state["seed_prosecution"] = prosecution_seed
                    st.session_state[user_key] = defence_arguments(
                        prosecutor_output=prosecution_seed,
                        user_case=case_text,
                    )
            except Exception as exc:
                st.exception(exc)
                return

    draft = st.text_area(
        f"Your {role} argument (editable)",
        value=st.session_state.get(user_key, ""),
        height=250,
        placeholder="Write or edit your courtroom argument here...",
    )

    reveal_working = st.checkbox("Show generated opposite-side reasoning", value=True)

    if st.button("Submit My Argument To Judge", type="primary", use_container_width=True):
        if not case_text:
            st.error("Please provide case text before submitting.")
            return
        if not draft.strip():
            st.error("Please write your argument before submitting.")
            return

        with st.spinner("Generating opposing side and judge order..."):
            try:
                if role == "Prosecution":
                    prosecutor_output = draft.strip()
                    defence_output = defence_arguments(
                        prosecutor_output=prosecutor_output,
                        user_case=case_text,
                    )
                else:
                    prosecutor_output = st.session_state.get("seed_prosecution") or prosecution_arguments(case_text)
                    defence_output = draft.strip()

                judge_output = judge_arguments(
                    prosecutor_output=prosecutor_output,
                    defence_output=defence_output,
                    user_case=case_text,
                )
            except Exception as exc:
                st.exception(exc)
                return

        if reveal_working:
            col1, col2 = st.columns(2)
            with col1:
                _show_outputs("Prosecution", prosecutor_output)
            with col2:
                _show_outputs("Defence", defence_output)
        _show_outputs("Judge Outcome", judge_output)

        payload = {
            "mode": "role_play",
            "selected_role": role.lower(),
            "prosecution": prosecutor_output,
            "defence": defence_output,
            "judge": judge_output,
        }
        st.download_button(
            "Download Session JSON",
            data=json.dumps(payload, ensure_ascii=False, indent=2),
            file_name="court_rag_role_play_result.json",
            mime="application/json",
            use_container_width=True,
        )


def main() -> None:
    _inject_styles()
    _header()

    st.sidebar.title("Session Mode")
    mode = st.sidebar.radio(
        "Choose experience",
        ["Role-Play", "Compare Both Sides"],
    )
    st.sidebar.caption("Tip: keep case input factual and concise for stronger legal reasoning output.")

    case_text = _render_case_input()
    if mode == "Role-Play":
        _role_play_mode(case_text)
    else:
        _compare_mode(case_text)


if __name__ == "__main__":
    main()
