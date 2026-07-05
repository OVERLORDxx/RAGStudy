import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure RAGStudy root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.document_processor import extract_text_from_pdf, chunk_document
from src.vector_store import LocalVectorStore
from src.rag_engine import RAGEngine
from src.question_generator import QuestionGenerator
from src.evaluator import RAGEvaluator

# 1. Page Configuration
st.set_page_config(
    page_title="RAGStudy ◆ Study Assistant & Exam Generator",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 2. Theme Toggle Pattern
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

IS_DARK = st.session_state.theme == "dark"

# 3. CSS Design System (Zinc Color Scheme & Minimal Chrome)
bg_color = "#09090b" if IS_DARK else "#ffffff"
bg_subtle = "#0c0c0f" if IS_DARK else "#f9fafb"
card_color = "#0c0c0f" if IS_DARK else "#ffffff"
card_hover = "#131316" if IS_DARK else "#f4f4f5"
border_color = "#1e1e24" if IS_DARK else "#e4e4e7"
border_subtle = "#16161a" if IS_DARK else "#f0f0f2"
text_color = "#fafafa" if IS_DARK else "#09090b"
text_muted = "#71717a"
text_dim = "#52525b" if IS_DARK else "#a1a1aa"
accent_color = "#2563eb"
shadow_val = "none" if IS_DARK else "0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03)"

custom_css = f"""
<style>
/* Hide Streamlit chrome */
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton,
div[data-testid="stSidebarCollapsedControl"] {{
    display: none !important;
}}

/* Global Styling */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {{
    background-color: {bg_color} !important;
    color: {text_color} !important;
    font-family: 'DM Sans', -apple-system, sans-serif !important;
}}
.block-container {{
    padding: 1.5rem 2rem 3rem !important;
    max-width: 1280px !important;
}}

/* Zinc Card Layouts */
.zinc-card {{
    background: {card_color};
    border: 1px solid {border_color};
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    box-shadow: {shadow_val};
    margin-bottom: 1.25rem;
    transition: background 0.2s ease, border-color 0.2s ease;
}}
.zinc-card:hover {{
    background: {card_hover};
    border-color: {accent_color};
}}

.zinc-card-title {{
    font-size: 0.95rem;
    font-weight: 600;
    color: {text_color};
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.zinc-card-desc {{
    font-size: 0.78rem;
    color: {text_muted};
    margin-bottom: 1rem;
}}

/* Columns Layout Gap */
[data-testid="stHorizontalBlock"] {{
    gap: 1.25rem !important;
}}

/* Custom Tabs styling */
button[data-baseweb="tab"] {{
    background: transparent !important;
    color: {text_muted} !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.2rem !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {text_color} !important;
    background: {card_color} !important;
    border-color: {border_color} !important;
}}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{
    display: none !important;
}}
[data-baseweb="tab-list"] {{
    gap: 6px !important;
    background: {bg_subtle} !important;
    border: 1px solid {border_color} !important;
    border-radius: 10px !important;
    padding: 4px !important;
    margin-bottom: 1.5rem !important;
}}

/* Metric Cards */
.metric-card {{
    background: {card_color};
    border: 1px solid {border_color};
    border-radius: 10px;
    padding: 1rem 1.25rem;
    box-shadow: {shadow_val};
    text-align: left;
}}
.metric-label {{
    font-size: 0.75rem;
    color: {text_muted};
    font-weight: 500;
    margin-bottom: 2px;
}}
.metric-value {{
    font-size: 1.6rem;
    font-weight: 700;
    color: {text_color};
    letter-spacing: -0.02em;
}}

/* Badges */
.badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 500;
}}
.badge-green {{
    color: {"#22c55e" if IS_DARK else "#16a34a"};
    background: {"rgba(34,197,94,0.12)" if IS_DARK else "rgba(22,163,74,0.08)"};
}}
.badge-red {{
    color: {"#ef4444" if IS_DARK else "#dc2626"};
    background: {"rgba(239,68,68,0.12)" if IS_DARK else "rgba(220,38,38,0.08)"};
}}
.badge-blue {{
    color: {accent_color};
    background: rgba(37,99,235,0.1);
}}
.badge-amber {{
    color: {"#f59e0b" if IS_DARK else "#d97706"};
    background: {"rgba(245,158,11,0.12)" if IS_DARK else "rgba(217,119,6,0.08)"};
}}

/* Citations & Source Chunks */
.citation-block {{
    border-left: 2px solid {accent_color};
    background: {bg_subtle};
    padding: 0.75rem 1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.8rem;
    margin-bottom: 0.75rem;
}}

/* Custom Scrollbar */
::-webkit-scrollbar {{
    width: 6px;
    height: 6px;
}}
::-webkit-scrollbar-track {{
    background: transparent;
}}
::-webkit-scrollbar-thumb {{
    background: {border_color};
    border-radius: 3px;
}}

/* Code blocks override */
code {{
    font-family: 'JetBrains Mono', monospace !important;
}}

/* Data Tables */
.data-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.8rem;
    margin-top: 0.5rem;
}}
.data-table th {{
    text-align: left;
    padding: 0.6rem 0.8rem;
    color: {text_muted};
    font-weight: 500;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-bottom: 1px solid {border_color};
}}
.data-table td {{
    padding: 0.65rem 0.8rem;
    color: {text_color};
    border-bottom: 1px solid {border_subtle};
}}
.data-table tr:last-child td {{
    border-bottom: none;
}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 4. Global Helper UI Functions
def metric_card(label, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#71717a" if not IS_DARK else "#a1a1aa", size=11),
    margin=dict(l=40, r=20, t=20, b=40),
    xaxis=dict(
        gridcolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)",
        zerolinecolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)",
        tickfont=dict(size=10, color="#71717a"),
    ),
    yaxis=dict(
        gridcolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)",
        zerolinecolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)",
        tickfont=dict(size=10, color="#71717a"),
    ),
)

# Initialize Session State
VECTOR_DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".vector_store"))
if not os.path.exists(VECTOR_DB_DIR):
    os.makedirs(VECTOR_DB_DIR)

# Load existing index if present
if "vector_store" not in st.session_state:
    v_store = LocalVectorStore()
    v_store.load(VECTOR_DB_DIR)
    st.session_state.vector_store = v_store

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "generated_exam" not in st.session_state:
    st.session_state.generated_exam = None

if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

# 5. Header Component
head_left, head_right = st.columns([9, 1.5])
with head_left:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.5rem;">
        <span style="font-size: 1.8rem; font-weight: 800; color: var(--text); letter-spacing: -0.04em;">◆ RAGStudy</span>
        <span class="badge badge-blue">v1.0.0</span>
        <span class="badge badge-amber" style="font-size:0.65rem;">RAG-Based Exam Generator</span>
    </div>
    """, unsafe_allow_html=True)
with head_right:
    theme_label = "☀️ Light Theme" if IS_DARK else "🌙 Dark Theme"
    st.button(theme_label, on_click=toggle_theme, use_container_width=True)

st.markdown("<hr style='margin: 0.5rem 0 1.25rem 0; border: none; border-top: 1px solid var(--border);'>", unsafe_allow_html=True)

# 6. Global Configuration Bar (Bypasses API Key propagation bugs)
api_col1, api_col2, api_col3 = st.columns([2, 5, 2.5])
with api_col1:
    llm_provider = st.selectbox("LLM Provider", ["Gemini", "OpenAI"], index=0, help="Select which model API provider to use.")
with api_col2:
    # Try to load API key from environment
    default_key = os.getenv("GEMINI_API_KEY") if llm_provider == "Gemini" else os.getenv("OPENAI_API_KEY")
    if not default_key:
        default_key = ""
    api_key_input = st.text_input(
        f"{llm_provider} API Key",
        value=default_key,
        type="password",
        help="Paste your API key here. Bypasses client SDK errors dynamically."
    )
with api_col3:
    # Vector store status indicator
    index_loaded = st.session_state.vector_store.index is not None
    chunks_count = len(st.session_state.vector_store.metadata) if index_loaded else 0
    status_label = "Connected & Active" if index_loaded else "No Index Loaded"
    status_class = "badge-green" if index_loaded else "badge-red"
    st.markdown(f"""
    <div style="margin-top: 4px;">
        <span style="font-size: 0.72rem; color: var(--text-muted); font-weight: 500; display:block; margin-bottom:4px;">Vector Database Status</span>
        <span class="badge {status_class}">{status_label} ({chunks_count} Chunks)</span>
    </div>
    """, unsafe_allow_html=True)

# Main Navigation
tab_upload, tab_chat, tab_exam, tab_eval = st.tabs([
    "📂 Document Ingestion",
    "💬 Study Assistant",
    "📝 Exam Paper Generator",
    "📊 System Evaluation"
])

# ----------------------------------------------------
# TAB 1: DOCUMENT INGESTION
# ----------------------------------------------------
with tab_upload:
    st.markdown("""
    <div class="zinc-card-title">📂 Upload and Index Lectures / Textbooks</div>
    <div class="zinc-card-desc">Provide a PDF file to process. The system will extract, chunk, embed, and index the text into a local FAISS database.</div>
    """, unsafe_allow_html=True)
    
    col_up1, col_up2 = st.columns([7, 3.5])
    with col_up1:
        uploaded_file = st.file_uploader("Choose a PDF document", type=["pdf"])
        
        # Ingestion configurations
        with st.expander("Advanced Chunking Settings", expanded=False):
            chunk_size = st.number_input("Chunk Size (words)", min_value=50, max_value=1000, value=150, step=50)
            chunk_overlap = st.number_input("Chunk Overlap (words)", min_value=10, max_value=200, value=30, step=10)
            
        ingest_btn = st.button("Extract and Build Vector Index", type="primary", use_container_width=True)
        
        if ingest_btn:
            if not uploaded_file:
                st.error("Please upload a PDF file first!")
            elif not api_key_input:
                st.warning("Please provide an API key in the global settings above.")
            else:
                with st.spinner("Processing PDF document... (Extracting pages)"):
                    try:
                        # Create temporary upload path
                        temp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "scratch/uploads"))
                        if not os.path.exists(temp_dir):
                            os.makedirs(temp_dir)
                        temp_filepath = os.path.join(temp_dir, uploaded_file.name)
                        
                        with open(temp_filepath, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                            
                        # Extract pages text
                        pages_data = extract_text_from_pdf(temp_filepath)
                        st.info(f"Successfully extracted {len(pages_data)} pages from PDF.")
                        
                        # Chunking
                        with st.spinner("Splitting text into overlapping semantic chunks..."):
                            chunks = chunk_document(pages_data, doc_name=uploaded_file.name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                            st.info(f"Splitting generated {len(chunks)} chunks.")
                            
                        # Build Index
                        with st.spinner("Generating local sentence embeddings & building FAISS index..."):
                            v_store = LocalVectorStore()
                            v_store.build_index(chunks)
                            v_store.save(VECTOR_DB_DIR)
                            st.session_state.vector_store = v_store
                            
                        st.success(f"Success! Vector database created and saved locally with {len(chunks)} chunks.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to ingest document: {e}")
                        
    with col_up2:
        # Document Stats Card
        st.markdown("<div class='zinc-card-title'>📊 Document Summary</div>", unsafe_allow_html=True)
        if index_loaded:
            chunks = st.session_state.vector_store.metadata
            doc_names = list(set([c["doc_name"] for c in chunks]))
            total_pages = max([c["page"] for c in chunks]) if chunks else 0
            
            c_stat1, c_stat2 = st.columns(2)
            with c_stat1:
                metric_card("Indexed Document", doc_names[0] if doc_names else "N/A")
                st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
                metric_card("Total Pages", total_pages)
            with c_stat2:
                metric_card("Total Chunks", len(chunks))
                st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
                metric_card("Avg Chunks/Page", f"{len(chunks)/max(1, total_pages):.1f}")
        else:
            st.info("No document has been ingested yet. Upload a PDF file to begin.")

# ----------------------------------------------------
# TAB 2: STUDY ASSISTANT (CHAT)
# ----------------------------------------------------
with tab_chat:
    st.markdown("""
    <div class="zinc-card-title">💬 Chat Assistant with Citation Verification</div>
    <div class="zinc-card-desc">Ask questions about your uploaded notes. Answers are traced to specific document chunks with interactive page citations.</div>
    """, unsafe_allow_html=True)
    
    if not index_loaded:
        st.info("⚠️ Please upload and index a PDF document under the **Document Ingestion** tab first!")
    else:
        # Chat setup
        chat_col1, chat_col2 = st.columns([6.5, 3.5])
        
        with chat_col1:
            # Query Input
            user_query = st.chat_input("Ask a question about the document...")
            
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "assistant" and "citations" in message:
                        # Draw styled citation tags
                        citation_tags = " ".join([f'<span class="badge badge-blue">Page {cit["page"]}</span>' for cit in message["citations"]])
                        st.markdown(f"<div style='margin-top: 8px;'><b>Sources cited:</b> {citation_tags}</div>", unsafe_allow_html=True)
            
            if user_query:
                # Append user query to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                
                with st.chat_message("user"):
                    st.markdown(user_query)
                    
                with st.chat_message("assistant"):
                    with st.spinner("Searching vector database & generating grounded response..."):
                        try:
                            # 1. Search Vector Store
                            search_results = st.session_state.vector_store.search(user_query, top_k=4)
                            retrieved_chunks = [item[0] for item in search_results]
                            
                            # 2. Call RAG Engine
                            rag_engine = RAGEngine(api_key=api_key_input, provider=llm_provider)
                            response = rag_engine.generate_response(user_query, retrieved_chunks)
                            
                            # Render answer
                            st.markdown(response["answer"])
                            
                            # Draw citation tags
                            if response["citations"]:
                                citation_tags = " ".join([f'<span class="badge badge-blue">Page {cit["page"]}</span>' for cit in response["citations"]])
                                st.markdown(f"<div style='margin-top: 8px;'><b>Sources cited:</b> {citation_tags}</div>", unsafe_allow_html=True)
                                
                            # Save to state
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": response["answer"],
                                "citations": response["citations"]
                            })
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to generate answer: {e}")
                            
        with chat_col2:
            st.markdown("<div class='zinc-card-title'>🔍 Citation Inspector</div>", unsafe_allow_html=True)
            # Inspect sources cited in the last assistant message
            last_message = next((msg for msg in reversed(st.session_state.chat_history) if msg["role"] == "assistant"), None)
            
            if last_message and "citations" in last_message and last_message["citations"]:
                st.markdown(f"<div class='zinc-card-desc'>Click to review the source text chunks for the last generated answer.</div>", unsafe_allow_html=True)
                for item in last_message["citations"]:
                    with st.expander(f"📖 Page {item['page']} (Chunk Reference)", expanded=True):
                        st.markdown(f"<div class='citation-block'>{item['text']}</div>", unsafe_allow_html=True)
            else:
                st.info("No sources have been cited yet. Ask a question to view citations here.")

# ----------------------------------------------------
# TAB 3: EXAM PAPER GENERATOR
# ----------------------------------------------------
with tab_exam:
    st.markdown("""
    <div class="zinc-card-title">📝 Question Paper Generator & Grounding Checker</div>
    <div class="zinc-card-desc">Generate structured academic exams from your materials. Each question's answer is run through an LLM-based faithfulness check.</div>
    """, unsafe_allow_html=True)
    
    if not index_loaded:
        st.info("⚠️ Please upload and index a PDF document under the **Document Ingestion** tab first!")
    else:
        # Form settings
        ex_col1, ex_col2 = st.columns([3, 7])
        
        with ex_col1:
            st.markdown("<div class='zinc-card-title'>⚙️ Generator Configuration</div>", unsafe_allow_html=True)
            num_q = st.slider("Number of Questions", min_value=1, max_value=10, value=4)
            diff_level = st.selectbox("Difficulty Level", ["Easy", "Medium", "Hard"])
            target_topic = st.text_input("Target Topic (Optional)", value="any", help="Specific topic or 'any'")
            
            gen_exam_btn = st.button("Generate Question Paper", type="primary", use_container_width=True)
            
            if gen_exam_btn:
                if not api_key_input:
                    st.warning("Please provide an API key in the global settings above.")
                else:
                    with st.spinner("Generating structured exam questions..."):
                        try:
                            # 1. Generate questions
                            chunks = st.session_state.vector_store.metadata
                            q_generator = QuestionGenerator(api_key=api_key_input, provider=llm_provider)
                            generated_questions = q_generator.generate_questions(
                                chunks, 
                                num_questions=num_q, 
                                difficulty=diff_level, 
                                topics=target_topic
                            )
                            
                            # 2. Run Grounding verification for each question
                            verified_questions = []
                            for q in generated_questions:
                                with st.spinner(f"Verifying grounding for: '{q['question'][:40]}...'"):
                                    is_grounded, explanation, cited_chunks = q_generator.verify_grounding(q, st.session_state.vector_store)
                                    q["grounded"] = is_grounded
                                    q["grounding_explanation"] = explanation
                                    q["grounding_cited"] = cited_chunks
                                    verified_questions.append(q)
                                    
                            st.session_state.generated_exam = verified_questions
                            st.success("Successfully generated and verified question paper!")
                        except Exception as e:
                            st.error(f"Failed to generate exam: {e}")
                            
        with ex_col2:
            st.markdown("<div class='zinc-card-title'>📄 Generated Exam Paper</div>", unsafe_allow_html=True)
            if not st.session_state.generated_exam:
                st.info("No exam paper has been generated yet. Configure settings and click 'Generate Question Paper'.")
            else:
                exam_questions = st.session_state.generated_exam
                
                # Question statistics
                total_qs = len(exam_questions)
                grounded_qs = sum([1 for q in exam_questions if q.get("grounded", False)])
                faithfulness_rate = (grounded_qs / total_qs) * 100 if total_qs > 0 else 0
                
                stat_col1, stat_col2 = st.columns(2)
                with stat_col1:
                    metric_card("Total Generated Questions", total_qs)
                with stat_col2:
                    metric_card("Grounded Faithfulness Rate", f"{faithfulness_rate:.1f}%")
                    
                st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
                
                for idx, q in enumerate(exam_questions):
                    badge_class = "badge-green" if q.get("grounded", False) else "badge-red"
                    badge_text = "Verified Grounded" if q.get("grounded", False) else "Not Grounded (Hallucination Risk)"
                    
                    st.markdown(f"""
                    <div class="zinc-card">
                        <div class="zinc-card-title">
                            Question {idx+1} <span class="badge badge-blue">{q['type']}</span>
                            <span class="badge {badge_class}" style="margin-left:auto;">{badge_text}</span>
                        </div>
                        <div style="font-size: 0.95rem; font-weight: 500; margin-bottom: 0.8rem;">{q['question']}</div>
                    """, unsafe_allow_html=True)
                    
                    # Show choices for MCQ
                    if q["type"] == "MCQ" and q.get("options"):
                        for option in q["options"]:
                            st.write(f"- {option}")
                            
                    # Answer Key Expander
                    with st.expander("🔑 View Expected Answer / Solution", expanded=False):
                        st.markdown(f"**Expected Answer:**\n{q['answer']}")
                        
                    # Grounding check Details Expander
                    with st.expander("🔍 Grounding Citation & Verification Detail", expanded=False):
                        st.markdown(f"**Verification status:** {badge_text}")
                        st.markdown(f"**LLM Assessment Explanation:**\n{q.get('grounding_explanation', 'No explanation provided.')}")
                        
                        if q.get("grounding_cited"):
                            st.markdown("**Cited Context Source Chunks:**")
                            for chunk in q["grounding_cited"]:
                                st.markdown(f"<div class='citation-block'><b>Page {chunk['page']}:</b><br>{chunk['text']}</div>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------
# TAB 4: SYSTEM EVALUATION
# ----------------------------------------------------
with tab_eval:
    st.markdown("""
    <div class="zinc-card-title">📊 Retrieval Evaluation Dashboard</div>
    <div class="zinc-card-desc">Measures index retrieval quality. The system auto-generates test QA pairs mapped to source chunks, queries the index, and plots Hit Rate @ K and Mean Reciprocal Rank (MRR).</div>
    """, unsafe_allow_html=True)
    
    if not index_loaded:
        st.info("⚠️ Please upload and index a PDF document under the **Document Ingestion** tab first!")
    else:
        ev_col1, ev_col2 = st.columns([3, 7])
        
        with ev_col1:
            st.markdown("<div class='zinc-card-title'>⚙️ Evaluation Settings</div>", unsafe_allow_html=True)
            eval_size = st.slider("Synthetic Dataset Size (Q&A)", min_value=2, max_value=10, value=4)
            top_k_eval = st.slider("Top K Retrieved Chunks", min_value=2, max_value=5, value=3)
            
            run_eval_btn = st.button("Run System Evaluation", type="primary", use_container_width=True)
            
            if run_eval_btn:
                if not api_key_input:
                    st.warning("Please provide an API key in the global settings above.")
                else:
                    with st.spinner("Generating synthetic evaluation set... (generating QA pairs from document)"):
                        try:
                            chunks = st.session_state.vector_store.metadata
                            evaluator = RAGEvaluator(api_key=api_key_input, provider=llm_provider)
                            
                            eval_set = evaluator.generate_evaluation_set(chunks, num_eval_pairs=eval_size)
                            st.info(f"Generated {len(eval_set)} synthetic QA evaluation pairs.")
                            
                            with st.spinner("Evaluating retrieval Hit Rate & Mean Reciprocal Rank..."):
                                results = evaluator.evaluate_retrieval(eval_set, st.session_state.vector_store, top_k=top_k_eval)
                                st.session_state.eval_results = results
                                st.success("Evaluation completed!")
                        except Exception as e:
                            st.error(f"Failed to run evaluation: {e}")
                            
        with ev_col2:
            st.markdown("<div class='zinc-card-title'>📈 Retrieval Metrics Dashboard</div>", unsafe_allow_html=True)
            if not st.session_state.eval_results:
                st.info("No evaluation data has been collected yet. Click 'Run System Evaluation' to start.")
            else:
                eval_data = st.session_state.eval_results
                
                # Metrics Row
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    metric_card(f"Hit Rate @ {top_k_eval}", f"{eval_data['hit_rate'] * 100:.1f}%")
                with m_col2:
                    metric_card(f"MRR @ {top_k_eval}", f"{eval_data['mrr']:.3f}")
                    
                # Plot charts
                st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
                
                # Chart card
                st.markdown(f"""
                <div class="zinc-card">
                    <div class="zinc-card-title">Hit Rate @ {top_k_eval} vs Fail Rate</div>
                    <div class="zinc-card-desc">Hit rate measures the percentage of test queries for which the correct source text chunk appears in the top retrieved list.</div>
                """, unsafe_allow_html=True)
                
                # Draw Plotly Pie chart
                labels = ['Hit', 'Miss']
                values = [eval_data['hit_rate'], 1.0 - eval_data['hit_rate']]
                colors = ['#22c55e', '#ef4444'] if IS_DARK else ['#16a34a', '#dc2626']
                
                fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, marker=dict(colors=colors))])
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Sans, sans-serif", color="#71717a" if not IS_DARK else "#a1a1aa"),
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=200
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Details Table
                st.markdown("<div class='zinc-card-title'>📋 Evaluation Test Logs</div>", unsafe_allow_html=True)
                table_rows = ""
                for idx, detail in enumerate(eval_data["details"]):
                    status_badge = f'<span class="badge badge-green">Hit</span>' if detail["hit"] == 1 else f'<span class="badge badge-red">Miss</span>'
                    table_rows += f"""
                    <tr>
                        <td>{idx+1}</td>
                        <td>{detail["question"]}</td>
                        <td>Page {detail["ground_truth_page"]}</td>
                        <td>{detail["rr"]:.3f}</td>
                        <td>{status_badge}</td>
                    </tr>
                    """
                    
                st.markdown(f"""
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Test Question</th>
                            <th>Gold Page</th>
                            <th>Reciprocal Rank</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
                """, unsafe_allow_html=True)
