import html
import os
import re
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import (
    extract_actionable_items,
    extract_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Styling
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --bg: #0e1217;
        --surface: #151b23;
        --surface-2: #1c242e;
        --border: #2a3441;
        --amber: #f5b544;
        --teal: #3fc1b0;
        --rose: #ef6f6c;
        --text: #e9edf2;
        --muted: #9aa7b5;
    }

    html, body, .stApp, [data-testid="stMarkdownContainer"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .stApp { background: var(--bg); color: var(--text); }
    .block-container { padding-top: 2rem; max-width: 1200px; }
    header[data-testid="stHeader"] { background: transparent; }

    [data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border);
    }
    h1, h2, h3, h4 {
        font-family: 'Bricolage Grotesque', sans-serif !important;
        letter-spacing: -0.01em;
    }

    /* ---------- Headings ---------- */
    .brand {
        font-family: 'Bricolage Grotesque', sans-serif;
        font-weight: 800;
        font-size: 1.5rem;
        line-height: 1.1;
        color: var(--text);
    }
    .brand span { color: var(--amber); }
    .brand-sub { color: var(--muted); font-size: .85rem; margin-top: .25rem; }

    .page-title {
        font-family: 'Bricolage Grotesque', sans-serif;
        font-weight: 800;
        font-size: clamp(1.9rem, 4.5vw, 3rem);
        line-height: 1.08;
        color: var(--text);
        margin: 0;
        overflow-wrap: anywhere;
    }
    .page-sub { color: var(--muted); margin-top: .5rem; max-width: 60ch; }

    /* ---------- Cards ---------- */
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.25rem 1.35rem;
        margin-bottom: 1rem;
    }
    .card.accent-amber { border-top: 3px solid var(--amber); }
    .card.accent-teal  { border-top: 3px solid var(--teal); }
    .card.accent-rose  { border-top: 3px solid var(--rose); }
    .card-title {
        font-family: 'Bricolage Grotesque', sans-serif;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: .75rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: .5rem;
    }
    .badge {
        font-family: 'IBM Plex Mono', monospace;
        font-size: .72rem;
        color: var(--muted);
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: .1rem .55rem;
    }
    .card-content {
        line-height: 1.7;
        max-width: 72ch;
        overflow-wrap: anywhere;
        white-space: pre-wrap;
    }
    .card ul { margin: 0; padding-left: 1.1rem; }
    .card li { margin-bottom: .5rem; line-height: 1.6; overflow-wrap: anywhere; }
    .card li::marker { color: var(--muted); }
    .empty-note { color: var(--muted); font-style: italic; }

    /* ---------- Stat tiles ---------- */
    .stat {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: .9rem 1rem;
    }
    .stat-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.5rem;
        font-weight: 500;
        color: var(--amber);
    }
    .stat-label { color: var(--muted); font-size: .85rem; }

    /* ---------- Pipeline track ---------- */
    .track {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: .5rem;
        margin: 1rem 0 1.5rem;
    }
    .track-step {
        background: var(--surface);
        border: 1px solid var(--border);
        border-bottom: 4px solid var(--border);
        border-radius: 10px;
        padding: .65rem .75rem;
        font-size: .85rem;
        color: var(--muted);
    }
    .track-step .t-icon { font-size: 1.1rem; display: block; margin-bottom: .2rem; }
    .track-step.done   { border-bottom-color: var(--teal); color: var(--text); }
    .track-step.active { border-bottom-color: var(--amber); color: var(--text); animation: pulse 1.4s ease-in-out infinite; }
    .track-step.error  { border-bottom-color: var(--rose); color: var(--rose); }
    @keyframes pulse { 50% { background: var(--surface-2); } }
    @media (prefers-reduced-motion: reduce) { .track-step.active { animation: none; } }
    @media (max-width: 800px) { .track { grid-template-columns: repeat(2, 1fr); } }

    /* ---------- Transcript ---------- */
    .transcript-box {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        max-height: 480px;
        overflow-y: auto;
        line-height: 1.8;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        max-width: 90ch;
    }
    .transcript-box mark {
        background: var(--amber);
        color: #1a1200;
        padding: 0 .15rem;
        border-radius: 3px;
    }

    /* ---------- Steps list (empty state) ---------- */
    .how {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    .how .card { margin: 0; }
    .how h4 { margin: 0 0 .35rem; font-size: 1rem; }
    .how p { margin: 0; color: var(--muted); font-size: .92rem; line-height: 1.55; }

    /* ---------- Streamlit widgets ---------- */
    /* Secondary (default) buttons */
    .stButton > button, .stDownloadButton > button {
        background: var(--surface-2);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 8px;
        font-weight: 500;
        transition: filter .15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        filter: brightness(1.15);
        color: var(--text);
        border-color: var(--amber);
    }
    /* Primary buttons */
    .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
        background: var(--amber);
        color: #1a1200;
        border: 0;
        font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {
        color: #1a1200;
    }
    .stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
        outline: 2px solid var(--teal);
        outline-offset: 2px;
    }
    .stTextInput input, .stTextArea textarea {
        background: var(--surface-2) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
    }
    .stTabs [data-baseweb="tab-list"] { gap: .25rem; border-bottom: 1px solid var(--border); }
    .stTabs [data-baseweb="tab"] { color: var(--muted); padding: .6rem 1rem; }
    .stTabs [aria-selected="true"] { color: var(--amber); }
    [data-testid="stChatMessage"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
    }
    hr { border-color: var(--border); }

    /* ---------- Sidebar text visibility ---------- */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] label p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stRadio"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] small,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: var(--text) !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: var(--muted) !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: var(--surface-2);
        border: 1px dashed var(--border);
        border-radius: 10px;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
        background: var(--surface);
        color: var(--text) !important;
        border: 1px solid var(--border);
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderFileName"] {
        color: var(--text) !important;
    }
    [data-testid="stSidebar"] input::placeholder {
        color: var(--muted) !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] .stButton > button p,
    [data-testid="stSidebar"] .stDownloadButton > button p {
        color: inherit !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] p,
    [data-testid="stSidebar"] .stDownloadButton > button[kind="primary"] p {
        color: #1a1200 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Session state
# ============================================================
defaults = {
    "result": None,
    "chat_history": [],
    "pipeline_steps": {},
    "pending_question": None,
    "elapsed": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

STEPS = [
    ("audio", "🔊", "Audio processing"),
    ("transcript", "📝", "Transcription"),
    ("title", "🏷️", "Title"),
    ("summary", "📋", "Summary"),
    ("extract", "🔍", "Extraction"),
    ("rag", "🧠", "Search index"),
]

SUGGESTED_QUESTIONS = [
    "What are the main points?",
    "Who is the intended audience?",
    "List any numbers or dates mentioned.",
]


# ============================================================
# Helpers
# ============================================================
def to_items(content):
    """Turn model output into a clean list of lines (strips bullets/numbering)."""
    if not content:
        return []
    if isinstance(content, (list, tuple)):
        lines = [str(c) for c in content]
    else:
        lines = str(content).splitlines()
    items = []
    for line in lines:
        cleaned = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", line).strip()
        if cleaned:
            items.append(cleaned)
    return items


def show_card(title, content, accent=None, as_list=False):
    """Render model output safely inside a styled card."""
    safe_title = html.escape(str(title))
    accent_cls = f" accent-{accent}" if accent else ""
    badge = ""

    if as_list:
        items = to_items(content)
        if items:
            badge = f'<span class="badge">{len(items)}</span>'
            body = "<ul>" + "".join(f"<li>{html.escape(i)}</li>" for i in items) + "</ul>"
        else:
            body = '<div class="empty-note">Nothing found in this video.</div>'
    else:
        text = str(content).strip() if content else ""
        body = (
            f'<div class="card-content">{html.escape(text)}</div>'
            if text
            else '<div class="empty-note">Nothing found in this video.</div>'
        )

    st.markdown(
        f"""
        <div class="card{accent_cls}">
            <div class="card-title"><span>{safe_title}</span>{badge}</div>
            {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_stat(value, label):
    st.markdown(
        f"""
        <div class="stat">
            <div class="stat-value">{html.escape(str(value))}</div>
            <div class="stat-label">{html.escape(label)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline(slot):
    """Draw the horizontal pipeline track into a placeholder."""
    steps = st.session_state.pipeline_steps
    if not steps:
        slot.empty()
        return
    icons = {"done": "✅", "active": "⏳", "error": "❌"}
    cells = ""
    for key, default_icon, label in STEPS:
        state = steps.get(key, "pending")
        icon = icons.get(state, default_icon)
        cells += (
            f'<div class="track-step {state}">'
            f'<span class="t-icon">{icon}</span>{html.escape(label)}</div>'
        )
    slot.markdown(f'<div class="track">{cells}</div>', unsafe_allow_html=True)


def highlight(text, query):
    """Escape text, then wrap matches of query in <mark>. Returns (html, count)."""
    safe = html.escape(text)
    if not query:
        return safe, 0
    pattern = re.compile(re.escape(html.escape(query)), re.IGNORECASE)
    matches = len(pattern.findall(safe))
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", safe), matches


def build_report(result):
    def bullets(content):
        items = to_items(content)
        return "\n".join(f"- {i}" for i in items) if items else "_None_"

    return (
        f"# {result['title']}\n\n"
        f"## Summary\n{result['summary']}\n\n"
        f"## Action items\n{bullets(result['action_items'])}\n\n"
        f"## Key decisions\n{bullets(result['key_decisions'])}\n\n"
        f"## Open questions\n{bullets(result['open_questions'])}\n\n"
        f"## Transcript\n{result['transcript']}\n"
    )


def answer_question(question):
    """Add a question to the chat, get an answer, and store both."""
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the transcript..."):
            try:
                answer = str(ask_question(st.session_state.result["rag_chain"], question))
                st.markdown(answer)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as exc:
                st.error(f"Couldn't get an answer: {exc}. Try rephrasing the question.")


def run_analysis(source, pipeline_slot):
    """Run the same processing sequence as the project's main.py."""
    steps = st.session_state.pipeline_steps
    progress = st.progress(0, text="Starting analysis...")
    started = time.time()

    def advance(key, pct, text):
        steps[key] = "done"
        render_pipeline(pipeline_slot)
        progress.progress(pct, text=text)

    def begin(key):
        steps[key] = "active"
        render_pipeline(pipeline_slot)

    try:
        begin("audio")
        chunks = process_input(source)
        if not chunks:
            raise ValueError("No audio chunks were generated.")
        advance("audio", 15, "Audio processed")

        begin("transcript")
        transcript = transcribe_all(chunks)
        if not transcript or not transcript.strip():
            raise ValueError("The transcript is empty. Check that the file has speech.")
        advance("transcript", 35, "Transcription complete")

        begin("title")
        title = generate_title(transcript)
        advance("title", 45, "Title generated")

        begin("summary")
        summary = summarize(transcript)
        advance("summary", 60, "Summary generated")

        begin("extract")
        action_items = extract_actionable_items(transcript)
        decisions = extract_decisions(transcript)
        questions = extract_questions(transcript)
        advance("extract", 75, "Insights extracted")

        begin("rag")
        rag_chain = build_rag_chain(transcript)
        advance("rag", 100, "Search index ready")

        st.session_state.result = {
            "title": title,
            "transcript": transcript,
            "summary": summary,
            "action_items": action_items,
            "key_decisions": decisions,
            "open_questions": questions,
            "rag_chain": rag_chain,
        }
        st.session_state.chat_history = []
        st.session_state.elapsed = time.time() - started
        st.toast("Analysis complete", icon="✅")

    except Exception as exc:
        for key, _, _ in STEPS:
            if steps.get(key) == "active":
                steps[key] = "error"
        render_pipeline(pipeline_slot)
        st.error(f"Analysis failed: {exc}")
        with st.expander("Technical details"):
            st.exception(exc)

    finally:
        progress.empty()


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown(
        '<div class="brand">🎬 Video <span>Assistant</span></div>'
        '<div class="brand-sub">Transcribe, summarise and ask questions.</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    input_mode = st.radio("Where is your video?", ["YouTube URL", "Upload a file"])

    source = ""
    uploaded_file = None

    if input_mode == "YouTube URL":
        source = st.text_input(
            "YouTube link",
            placeholder="https://www.youtube.com/watch?v=...",
        ).strip()
    else:
        uploaded_file = st.file_uploader(
            "Video or audio file",
            type=["mp4", "mov", "mkv", "webm", "mp3", "wav", "m4a", "mpeg"],
            help="Supported: mp4, mov, mkv, webm, mp3, wav, m4a, mpeg",
        )
        if uploaded_file:
            size_mb = uploaded_file.size / (1024 * 1024)
            st.caption(f"{uploaded_file.name} · {size_mb:.1f} MB")

    analyse_clicked = st.button("Analyse video", use_container_width=True, type="primary")

    if st.session_state.result:
        st.divider()
        st.download_button(
            "Download report (.md)",
            data=build_report(st.session_state.result),
            file_name="video_report.md",
            mime="text/markdown",
            use_container_width=True,
            type="primary",
        )
        if st.button("Start new analysis", use_container_width=True, type="secondary"):
            st.session_state.result = None
            st.session_state.chat_history = []
            st.session_state.pipeline_steps = {}
            st.session_state.elapsed = None
            st.rerun()


# ============================================================
# Main page
# ============================================================
result = st.session_state.result

if result:
    st.markdown(f'<h1 class="page-title">{html.escape(str(result["title"]))}</h1>', unsafe_allow_html=True)
else:
    st.markdown('<h1 class="page-title">Turn any video into searchable notes</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">Paste a YouTube link or upload a recording. '
        "You get a transcript, a summary, action items, and a chat that answers from the video.</p>",
        unsafe_allow_html=True,
    )

pipeline_slot = st.empty()
render_pipeline(pipeline_slot)

# ---------- Start pipeline ----------
if analyse_clicked:
    if input_mode == "YouTube URL":
        if not source:
            st.error("Enter a YouTube link in the sidebar.")
        else:
            st.session_state.pipeline_steps = {k: "pending" for k, _, _ in STEPS}
            st.session_state.result = None
            st.session_state.chat_history = []
            run_analysis(source, pipeline_slot)
            if st.session_state.result:
                st.rerun()

    elif uploaded_file is None:
        st.error("Upload a video or audio file in the sidebar.")
    else:
        temp_path = None
        try:
            suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name

            st.session_state.pipeline_steps = {k: "pending" for k, _, _ in STEPS}
            st.session_state.result = None
            st.session_state.chat_history = []
            run_analysis(temp_path, pipeline_slot)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
        if st.session_state.result:
            st.rerun()

# ---------- Results ----------
result = st.session_state.result

if result:
    words = len(result["transcript"].split())
    read_min = max(1, round(words / 200))
    n_actions = len(to_items(result["action_items"]))
    n_decisions = len(to_items(result["key_decisions"]))
    n_questions = len(to_items(result["open_questions"]))

    s1, s2, s3, s4 = st.columns(4, gap="small")
    with s1:
        show_stat(f"{words:,}", "words transcribed")
    with s2:
        show_stat(f"{read_min} min", "reading time")
    with s3:
        show_stat(n_actions + n_decisions + n_questions, "insights found")
    with s4:
        elapsed = st.session_state.elapsed
        show_stat(f"{elapsed:.0f}s" if elapsed else "–", "processing time")

    st.write("")
    tab_overview, tab_transcript, tab_chat = st.tabs(
        ["Overview", "Transcript", "Ask the video"]
    )

    # ----- Overview -----
    with tab_overview:
        show_card("Summary", result["summary"], accent="amber")
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            show_card("Action items", result["action_items"], accent="teal", as_list=True)
        with c2:
            show_card("Key decisions", result["key_decisions"], accent="amber", as_list=True)
        with c3:
            show_card("Open questions", result["open_questions"], accent="rose", as_list=True)

    # ----- Transcript -----
    with tab_transcript:
        search_col, dl_col = st.columns([4, 1], gap="medium")
        with search_col:
            query = st.text_input(
                "Search transcript",
                placeholder="Type a word or phrase",
                label_visibility="collapsed",
            ).strip()
        with dl_col:
            st.download_button(
                "Download .txt",
                data=result["transcript"],
                file_name="transcript.txt",
                mime="text/plain",
                use_container_width=True,
                type="primary",
            )

        body, matches = highlight(result["transcript"], query)
        if query:
            if matches:
                st.caption(f"{matches} match{'es' if matches != 1 else ''} for “{query}”")
            else:
                st.caption(f"No matches for “{query}”")
        st.markdown(f'<div class="transcript-box">{body}</div>', unsafe_allow_html=True)

    # ----- Chat -----
    with tab_chat:
        st.caption("Answers come from the transcript only.")

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if not st.session_state.chat_history:
            st.markdown("**Try asking**")
            chip_cols = st.columns(len(SUGGESTED_QUESTIONS))
            for col, suggestion in zip(chip_cols, SUGGESTED_QUESTIONS):
                if col.button(suggestion, key=f"chip_{suggestion}", type="secondary", use_container_width=True):
                    st.session_state.pending_question = suggestion
                    st.rerun()

        if st.session_state.pending_question:
            queued = st.session_state.pending_question
            st.session_state.pending_question = None
            answer_question(queued)

        typed = st.chat_input("Ask something about this video")
        if typed:
            answer_question(typed)

        if st.session_state.chat_history:
            if st.button("Clear chat", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

elif not st.session_state.pipeline_steps:
    st.markdown(
        """
        <div class="how">
            <div class="card accent-amber">
                <h4>1. Add a video</h4>
                <p>Paste a YouTube link or upload an mp4, mov, mkv, webm, mp3, wav or m4a file.</p>
            </div>
            <div class="card accent-teal">
                <h4>2. Let it process</h4>
                <p>The video is transcribed, summarised and scanned for action items, decisions and open questions.</p>
            </div>
            <div class="card accent-rose">
                <h4>3. Read or ask</h4>
                <p>Skim the overview, search the transcript, or ask questions and get answers from the video.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
