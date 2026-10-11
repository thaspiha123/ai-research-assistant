
import hashlib
import io

import streamlit as st
from gtts import gTTS
from streamlit_mic_recorder import speech_to_text

from pdf_utils import extract_text_from_pdf
from chunking import chunk_pages
from vector_store import create_vector_store
from rag_pipeline import answer_question

from features import (
    generate_summary,
    generate_research_gaps,
    explain_paper_simply,
    generate_smart_notes,
    generate_quiz,
    generate_interactive_quiz,
    generate_research_mentor_response,
    compare_papers,
    find_citations,
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    .stApp {
        background-color: #f5f7fb;
        color: #19263b;
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background-color: rgba(245, 247, 251, 0.95);
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e4e9f2;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: #14213d !important;
        letter-spacing: -0.5px;
    }

    h1 {
        font-size: 2.25rem !important;
        font-weight: 800 !important;
    }

    h2 {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }

    h3 {
        font-size: 1.12rem !important;
        font-weight: 700 !important;
    }

    p, label, li {
        color: #46546b;
        line-height: 1.7;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1450px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }

    .hero {
        background: #ffffff;
        border: 1px solid #e3e9f3;
        border-radius: 20px;
        padding: 30px 34px;
        margin-bottom: 25px;
        box-shadow: 0 5px 24px rgba(25, 43, 77, 0.035);
    }

    .eyebrow {
        color: #4169d8;
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 9px;
    }

    .hero-title {
        color: #14213d;
        font-family: 'Manrope', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.25;
        margin-bottom: 10px;
    }

    .hero-description {
        color: #64728a;
        font-size: 1rem;
        max-width: 760px;
        margin-bottom: 0;
    }

    .section-card {
        background: #ffffff;
        border: 1px solid #e3e9f3;
        border-radius: 16px;
        padding: 22px 24px;
        margin: 12px 0 20px 0;
        box-shadow: 0 4px 18px rgba(25, 43, 77, 0.025);
    }

    .section-label {
        color: #77849b;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 8px;
    }

    .document-name {
        color: #172b4d;
        font-size: 1.05rem;
        font-weight: 700;
        overflow-wrap: anywhere;
    }

    .status-pill {
        display: inline-block;
        background: #e9f7ef;
        color: #217346;
        border: 1px solid #ccebd8;
        border-radius: 20px;
        padding: 5px 11px;
        font-size: 0.77rem;
        font-weight: 600;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        background: #3159c9;
        color: #ffffff;
        border: 1px solid #3159c9;
        border-radius: 9px;
        min-height: 43px;
        padding: 0.55rem 1.15rem;
        font-weight: 600;
        transition: all 0.18s ease;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background: #2547a8;
        color: #ffffff;
        border-color: #2547a8;
    }

    .stButton > button:focus,
    .stFormSubmitButton > button:focus {
        box-shadow: 0 0 0 3px rgba(49, 89, 201, 0.18);
    }

    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input {
        background: #ffffff;
        border: 1px solid #dce3ee;
        border-radius: 9px;
        color: #19263b;
    }

    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #b9c8e2;
        border-radius: 14px;
        padding: 12px;
    }

    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 7px;
        border-bottom: 1px solid #e1e7f0;
        padding-bottom: 8px;
    }

    [data-testid="stTabs"] button[data-baseweb="tab"] {
        background: #edf1f8;
        border: 1px solid transparent;
        border-radius: 9px;
        color: #596981;
        padding: 10px 15px;
        font-weight: 600;
        white-space: nowrap;
    }

    [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
        background: #e7edff;
        color: #2547a8;
        border-color: #cdd9ff;
    }

    [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #e1e7f0;
        border-radius: 10px;
        overflow: hidden;
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    hr {
        border-color: #e4e9f2;
        margin: 24px 0;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e3e9f3;
        border-radius: 13px;
        padding: 16px 18px;
    }

    [data-testid="stMetricLabel"] {
        color: #718099;
    }

    [data-testid="stMetricValue"] {
        color: #1e3769;
    }

    .footer {
        text-align: center;
        color: #8a96aa;
        font-size: 0.8rem;
        border-top: 1px solid #e4e9f2;
        padding-top: 18px;
        margin-top: 40px;
    }

    @media (max-width: 768px) {
        .hero {
            padding: 22px;
        }

        .hero-title {
            font-size: 1.65rem;
        }

        h1 {
            font-size: 1.8rem !important;
        }

        .section-card {
            padding: 17px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "processed_files" not in st.session_state:
    st.session_state.processed_files = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "feature_results" not in st.session_state:
    st.session_state.feature_results = {}

if "pending_voice_question" not in st.session_state:
    st.session_state.pending_voice_question = ""

if "last_voice_audio" not in st.session_state:
    st.session_state.last_voice_audio = None

if "last_voice_audio_file" not in st.session_state:
    st.session_state.last_voice_audio_file = None


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0 5px 22px 5px;">
            <div style="font-size: 0.72rem; color: #4169d8;
                        font-weight: 700; letter-spacing: 1.5px;
                        text-transform: uppercase;">
                Research Workspace
            </div>
            <div style="font-size: 1.45rem; font-weight: 800;
                        color: #14213d; margin-top: 7px;
                        font-family: 'Manrope', sans-serif;">
                Research Assistant
            </div>
            <div style="font-size: 0.86rem; color: #78869d;
                        margin-top: 7px; line-height: 1.6;">
                Read, understand and explore research papers.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("#### Workspace overview")
    st.caption("Upload a paper to activate the research tools.")

    st.markdown("---")
    st.caption("Built with Streamlit, Python, and Mini RAG.")


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI-powered research workspace</div>
        <div class="hero-title">Understand research, faster.</div>
        <p class="hero-description">
            Upload a research paper to summarize findings, ask questions,
            generate study notes, discover research gaps, compare papers,
            and locate supporting passages.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# PDF UPLOAD AND PROCESSING
# --------------------------------------------------

st.markdown("### Your research document")

uploaded_file = st.file_uploader(
    "Upload a research paper in PDF format",
    type=["pdf"],
    key="main_pdf_upload",
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-label">Getting started</div>
            <div style="font-size: 1.15rem; font-weight: 700;
                        color: #172b4d; margin-bottom: 8px;">
                Add your first research paper
            </div>
            <div style="color: #68778e; line-height: 1.7;">
                Choose a readable PDF document using the upload area above.
                Once it is processed, your research tools will be ready.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

pdf_bytes = uploaded_file.getvalue()
file_hash = hashlib.sha256(pdf_bytes).hexdigest()

if file_hash not in st.session_state.processed_files:
    with st.spinner("Processing your research paper..."):
        try:
            pages = extract_text_from_pdf(uploaded_file)

            if not pages:
                st.error("No readable text was found in this PDF.")
                st.stop()

            chunks = chunk_pages(pages)

            if not chunks:
                st.error("Could not create text chunks from this PDF.")
                st.stop()

            index, chunks = create_vector_store(chunks)

            st.session_state.processed_files[file_hash] = {
                "pages": pages,
                "chunks": chunks,
                "index": index,
                "name": uploaded_file.name,
            }

        except Exception as error:
            st.error(f"Could not process the PDF: {error}")
            st.stop()

document = st.session_state.processed_files[file_hash]
pages = document["pages"]
chunks = document["chunks"]
index = document["index"]

st.markdown(
    f"""
    <div class="section-card">
        <div class="section-label">Active document</div>
        <div class="document-name">{document["name"]}</div>
        <div style="margin-top: 12px;">
            <span class="status-pill">Document processed</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_columns = st.columns(3)

with metric_columns[0]:
    st.metric("Pages processed", len(pages))

with metric_columns[1]:
    st.metric("Text chunks", len(chunks))

with metric_columns[2]:
    st.metric("Research tools", 9)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def get_feature_result(feature_name):
    key = f"{file_hash}_{feature_name}"
    return st.session_state.feature_results.get(key)


def save_feature_result(feature_name, result):
    key = f"{file_hash}_{feature_name}"
    st.session_state.feature_results[key] = result


def display_result(result):
    if isinstance(result, str):
        st.markdown(result)
    else:
        st.write(result)


def get_answer_result(question):
    result = answer_question(
        index=index,
        chunks=chunks,
        question=question.strip(),
    )

    if isinstance(result, dict):
        answer = result.get("answer", "")
        sources = result.get("sources", [])
    else:
        answer = str(result)
        sources = []

    if not answer or not str(answer).strip():
        raise ValueError("The AI returned an empty answer.")

    return str(answer), sources


def display_sources(sources):
    if not sources:
        return

    with st.expander("View supporting passages"):
        for source in sources:
            if isinstance(source, dict):
                page_number = source.get("page", "Unknown")
                source_text = source.get("text", "")
            else:
                page_number = "Unknown"
                source_text = str(source)

            st.markdown(f"**Page {page_number}**")
            st.write(source_text)


# --------------------------------------------------
# FEATURE NAVIGATION
# --------------------------------------------------

tabs = st.tabs([
    "Ask Questions",
    "Paper Summary",
    "Simple Explanation",
    "Smart Notes",
    "Research Gaps",
    "Quiz Generator",
    "Research Mentor",
    "Paper Comparison",
    "Citation Finder",
])


# --------------------------------------------------
# TAB 1: ASK QUESTIONS
# --------------------------------------------------

with tabs[0]:
    st.subheader("Ask questions about your paper")
    st.write(
        "Ask in your own words. Answers are generated using the "
        "uploaded document and relevant retrieved passages."
    )

    st.markdown("#### Voice question")

    spoken_question = speech_to_text(
        language="en",
        start_prompt="Start Recording",
        stop_prompt="Stop Recording",
        just_once=True,
        key="voice_question_input",
    )

    if spoken_question and spoken_question.strip():
        st.session_state.pending_voice_question = (
            spoken_question.strip()
        )

    saved_question = st.session_state.pending_voice_question

    if saved_question:
        st.markdown("**Recognized question**")
        st.info(saved_question)

        if st.button(
            "Get Voice Answer",
            key="get_voice_answer_button",
        ):
            try:
                with st.spinner("Finding an answer..."):
                    answer, sources = get_answer_result(
                        saved_question
                    )

                st.session_state.chat_history.append({
                    "file_hash": file_hash,
                    "question": saved_question,
                    "answer": answer,
                    "sources": sources,
                })

                st.markdown("#### Answer")
                st.markdown(answer)
                display_sources(sources)

                with st.spinner("Preparing spoken answer..."):
                    audio_buffer = io.BytesIO()
                    gTTS(text=answer, lang="en").write_to_fp(
                        audio_buffer
                    )
                    audio_buffer.seek(0)

                    audio_bytes = audio_buffer.getvalue()
                    st.session_state.last_voice_audio = audio_bytes
                    st.session_state.last_voice_audio_file = file_hash

                st.markdown("#### Spoken answer")
                st.audio(audio_bytes, format="audio/mp3")

                st.session_state.pending_voice_question = ""

            except Exception as error:
                st.error(f"Voice answer failed: {error}")

    if (
        st.session_state.last_voice_audio
        and st.session_state.last_voice_audio_file == file_hash
        and not st.session_state.pending_voice_question
    ):
        st.markdown("#### Last spoken answer")
        st.audio(
            st.session_state.last_voice_audio,
            format="audio/mp3",
        )

    st.divider()

    st.markdown("#### Text question")

    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "Enter your question",
            placeholder="What is the main objective of this paper?",
        )
        ask_clicked = st.form_submit_button("Get Answer")

    if ask_clicked:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Finding an answer..."):
                    answer, sources = get_answer_result(question)

                st.session_state.chat_history.append({
                    "file_hash": file_hash,
                    "question": question.strip(),
                    "answer": answer,
                    "sources": sources,
                })

            except Exception as error:
                st.error(f"Could not generate an answer: {error}")

    current_history = [
        item
        for item in st.session_state.chat_history
        if item["file_hash"] == file_hash
    ]

    if current_history:
        st.markdown("#### Conversation history")

        for item in reversed(current_history):
            with st.container(border=True):
                st.markdown("**Question**")
                st.write(item["question"])
                st.markdown("**Answer**")
                st.markdown(item["answer"])
                display_sources(item.get("sources", []))


# --------------------------------------------------
# TAB 2: PAPER SUMMARY
# --------------------------------------------------

with tabs[1]:
    st.subheader("Paper Summary")
    st.write(
        "Create a concise summary of the main ideas and findings."
    )

    if st.button("Generate Summary", key="summary_button"):
        try:
            with st.spinner("Generating summary..."):
                result = generate_summary(pages)
            save_feature_result("summary", result)
        except Exception as error:
            st.error(f"Summary generation failed: {error}")

    summary = get_feature_result("summary")

    if summary is not None:
        with st.container(border=True):
            st.markdown("#### Summary")
            display_result(summary)


# --------------------------------------------------
# TAB 3: SIMPLE EXPLANATION
# --------------------------------------------------

with tabs[2]:
    st.subheader("Simple Explanation")
    st.write(
        "Rewrite difficult research concepts in easier language."
    )

    if st.button(
        "Explain Paper Simply",
        key="simple_explanation_button",
    ):
        try:
            with st.spinner("Preparing a simple explanation..."):
                result = explain_paper_simply(pages)
            save_feature_result("simple_explanation", result)
        except Exception as error:
            st.error(f"Simple explanation failed: {error}")

    explanation = get_feature_result("simple_explanation")

    if explanation is not None:
        with st.container(border=True):
            st.markdown("#### Explanation")
            display_result(explanation)


# --------------------------------------------------
# TAB 4: SMART NOTES
# --------------------------------------------------

with tabs[3]:
    st.subheader("Smart Notes Generator")
    st.write(
        "Generate structured notes that are easier to review and study."
    )

    if st.button(
        "Generate Smart Notes",
        key="smart_notes_button",
    ):
        try:
            with st.spinner("Generating smart notes..."):
                result = generate_smart_notes(pages)
            save_feature_result("smart_notes", result)
        except Exception as error:
            st.error(f"Smart notes generation failed: {error}")

    notes = get_feature_result("smart_notes")

    if notes is not None:
        with st.container(border=True):
            st.markdown("#### Generated Notes")
            display_result(notes)

        st.download_button(
            "Download Smart Notes",
            data=notes if isinstance(notes, str) else str(notes),
            file_name="smart_notes.txt",
            mime="text/plain",
            key="download_smart_notes",
        )


# --------------------------------------------------
# TAB 5: RESEARCH GAP FINDER
# --------------------------------------------------

with tabs[4]:
    st.subheader("Research Gap Finder")
    st.write(
        "Explore possible limitations, research gaps, and future directions."
    )

    if st.button(
        "Find Research Gaps",
        key="research_gaps_button",
    ):
        try:
            with st.spinner("Analyzing the paper..."):
                result = generate_research_gaps(pages)
            save_feature_result("research_gaps", result)
        except Exception as error:
            st.error(f"Research gap analysis failed: {error}")

    gaps = get_feature_result("research_gaps")

    if gaps is not None:
        with st.container(border=True):
            st.markdown("#### Analysis Results")
            display_result(gaps)


# --------------------------------------------------
# TAB 6: QUIZ GENERATOR
# --------------------------------------------------

with tabs[5]:
    st.subheader("Quiz Generator")
    st.write(
        "Test your understanding of the research paper."
    )

    quiz_type = st.radio(
        "Choose a quiz type",
        ["Study Quiz", "Interactive Practice Quiz"],
        key="quiz_type",
        horizontal=True,
    )

    if quiz_type == "Study Quiz":
        number_of_questions = st.number_input(
            "Number of questions",
            min_value=1,
            max_value=20,
            value=10,
            step=1,
            key="study_quiz_number",
        )

        if st.button(
            "Generate Quiz",
            key="generate_quiz_button",
        ):
            try:
                with st.spinner("Creating your quiz..."):
                    result = generate_quiz(
                        pages,
                        number=int(number_of_questions),
                    )
                save_feature_result("study_quiz", result)
            except Exception as error:
                st.error(f"Quiz generation failed: {error}")

        quiz = get_feature_result("study_quiz")

        if quiz is not None:
            with st.container(border=True):
                st.markdown("#### Study Quiz")
                display_result(quiz)

            st.download_button(
                "Download Quiz",
                data=quiz if isinstance(quiz, str) else str(quiz),
                file_name="research_paper_quiz.txt",
                mime="text/plain",
                key="download_study_quiz",
            )

    else:
        practice_count = st.number_input(
            "Number of practice questions",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            key="practice_quiz_number",
        )

        if st.button(
            "Generate Practice Quiz",
            key="interactive_quiz_button",
        ):
            try:
                with st.spinner("Creating your practice quiz..."):
                    result = generate_interactive_quiz(
                        pages,
                        number=int(practice_count),
                    )
                save_feature_result("interactive_quiz", result)
            except Exception as error:
                st.error(f"Practice quiz generation failed: {error}")

        practice_quiz = get_feature_result("interactive_quiz")

        if practice_quiz is not None:
            with st.container(border=True):
                st.markdown("#### Practice Quiz")
                display_result(practice_quiz)


# --------------------------------------------------
# TAB 7: AI RESEARCH MENTOR
# --------------------------------------------------

with tabs[6]:
    st.subheader("AI Research Mentor")
    st.write(
        "Ask for guidance on understanding or extending your research."
    )

    mentor_question = st.text_input(
        "Your question",
        placeholder="How could this research be extended?",
        key="mentor_question",
    )

    if st.button(
        "Ask Research Mentor",
        key="mentor_button",
    ):
        if not mentor_question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Preparing mentor guidance..."):
                    result = generate_research_mentor_response(
                        pages,
                        mentor_question.strip(),
                    )

                save_feature_result("mentor_latest", result)

            except Exception as error:
                st.error(f"Research mentor failed: {error}")

    mentor_result = get_feature_result("mentor_latest")

    if mentor_result is not None:
        with st.container(border=True):
            st.markdown("#### Mentor Response")
            display_result(mentor_result)


# --------------------------------------------------
# TAB 8: PAPER COMPARISON
# --------------------------------------------------

with tabs[7]:
    st.subheader("Paper Comparison")
    st.write(
        "Upload another research paper to compare its content "
        "with the active document."
    )

    second_pdf = st.file_uploader(
        "Upload the second research paper",
        type=["pdf"],
        key="comparison_pdf_upload",
    )

    if second_pdf is not None:
        if second_pdf.getvalue() == pdf_bytes:
            st.warning("Please upload a different research paper.")
        elif st.button(
            "Compare Papers",
            key="compare_papers_button",
        ):
            try:
                with st.spinner("Comparing both papers..."):
                    second_pages = extract_text_from_pdf(second_pdf)

                    if not second_pages:
                        st.error(
                            "No readable text was found in the second PDF."
                        )
                    else:
                        result = compare_papers([
                            {
                                "name": document["name"],
                                "pages": pages,
                            },
                            {
                                "name": second_pdf.name,
                                "pages": second_pages,
                            },
                        ])

                        save_feature_result("paper_comparison", result)

            except Exception as error:
                st.error(f"Paper comparison failed: {error}")

    comparison = get_feature_result("paper_comparison")

    if comparison is not None:
        with st.container(border=True):
            st.markdown("#### Comparison Results")
            display_result(comparison)


# --------------------------------------------------
# TAB 9: CITATION FINDER
# --------------------------------------------------

with tabs[8]:
    st.subheader("Citation Finder")
    st.write(
        "Find passages relevant to a topic or question and view "
        "their original page numbers."
    )

    citation_question = st.text_input(
        "What information do you want to find?",
        placeholder="Find passages about the research methodology.",
        key="citation_question",
    )

    if st.button(
        "Find Supporting Passages",
        key="find_citations_button",
    ):
        if not citation_question.strip():
            st.warning("Please enter a topic or question.")
        else:
            try:
                with st.spinner("Searching the document..."):
                    citations = find_citations(
                        index=index,
                        chunks=chunks,
                        question=citation_question.strip(),
                        top_k=5,
                    )

                save_feature_result("citations_latest", citations)

            except Exception as error:
                st.error(f"Citation search failed: {error}")

    citations = get_feature_result("citations_latest")

    if citations is not None:
        if not citations:
            st.info("No relevant passages were found.")
        else:
            st.markdown("#### Relevant Passages")

            for number, citation in enumerate(citations, start=1):
                with st.container(border=True):
                    st.markdown(
                        f"**Passage {number}**"
                    )
                    st.caption(
                        f"Original page: {citation.get('page', 'Unknown')}"
                    )
                    st.write(citation.get("text", ""))

    st.caption(
        "This feature retrieves relevant passages and page numbers "
        "from the uploaded paper. It does not independently verify "
        "a complete APA or IEEE bibliography."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        AI Research Assistant using Mini RAG
    </div>
    """,
    unsafe_allow_html=True,
)
