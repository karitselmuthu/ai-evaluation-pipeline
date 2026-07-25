import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Make the project root importable when this script is launched directly
# (e.g. `streamlit run src/dashboard/app.py`). Streamlit puts src/dashboard/ on
# sys.path, not the project root, so `from src.core...` fails without this.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Load env configurations
load_dotenv()

# Set page config for premium look
st.set_page_config(
    page_title="AI Evaluation Pipeline Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling via CSS (sleek dark theme with gradients and glassmorphism cards)
st.markdown("""
<style>
    /* CSS System Design Variable Tokens */
    :root {
        --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        --card-bg: rgba(30, 41, 59, 0.7);
        --card-border: rgba(255, 255, 255, 0.08);
        --text-muted: #94a3b8;
    }
    
    /* Main Background & Fonts */
    .stApp {
        background: radial-gradient(circle at top right, #0f172a, #020617);
        color: #f8fafc;
        font-family: 'Outfit', 'Inter', -apple-system, sans-serif;
    }
    
    /* Sleek Typography */
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    
    .gradient-text {
        background: linear-gradient(135deg, #a5b4fc 0%, #e879f9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* KPI Card styling */
    .kpi-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 30px -15px rgba(0,0,0,0.3);
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .kpi-val {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(135deg, #f8fafc, #cbd5e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .kpi-label {
        font-size: 14px;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Imports from core pipeline
from src.core.dataset import DatasetLoader
from src.core.generator import LLMGenerator
from src.core.evaluator import EvaluationEngine
from src.core.aggregator import ScoreAggregator
from src.core.secret_bootstrap import bootstrap_secrets
from src.core.log_filters import install_noise_filter

# Drop the Antigravity/Gemini harness 429 spend-cap storm from the Streamlit
# terminal (set VERBOSE_HARNESS_LOGS=1 to restore it). Idempotent.
install_noise_filter()

# Resolve LLM keys via the SecretProvider seam (vault or env) before the
# dashboard constructs any generator / judge. Idempotent across Streamlit reruns.
bootstrap_secrets()

# Directory checks
OUTPUT_DIR = Path("output")
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

# Header block
st.markdown("<h1>⚡ AI <span class='gradient-text'>Evaluation Pipeline Hub</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8; font-size:16px; margin-top:-10px;'>Verify model quality, accuracy, safety, and performance dimensions in real-time.</p>", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("### ⚙️ Pipeline Configuration")

# Dataset Selector
datasets = [str(p) for p in Path(".").glob("tests/data/*.json")] + [str(p) for p in UPLOAD_DIR.glob("*.json")]
if not datasets:
    datasets = ["tests/data/sample_dataset.json"]

selected_dataset = st.sidebar.selectbox("Select Dataset", datasets)

# Upload New Dataset
uploaded_file = st.sidebar.file_uploader("Upload custom JSON dataset", type=["json"])
if uploaded_file is not None:
    uploaded_path = UPLOAD_DIR / uploaded_file.name
    with open(uploaded_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"Uploaded: {uploaded_file.name}")
    st.rerun()

# Model Configurations (routed through the local vault proxy).
# "default" lets the proxy pick / rotate providers (generation only).
model_under_test = st.sidebar.selectbox(
    "Application Model",
    ["default", "llama-3.3-70b-versatile", "mistral-small-latest"],
    index=0
)

# Judge is pinned: "default" is not allowed (it would let the proxy switch
# graders mid-run and break score comparability).
judge_model = st.sidebar.selectbox(
    "Evaluation Judge Model",
    ["llama-3.3-70b-versatile", "mistral-small-latest"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 System Prompt Template")
prompt_template_str = st.sidebar.text_area(
    "System Instruction Template",
    value=LLMGenerator.DEFAULT_PROMPT_TEMPLATE,
    height=200
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Execution")

run_eval_button = st.sidebar.button("🚀 Execute Standard Run", use_container_width=True)

# ----------------- STANDARD PIPELINE EXECUTION -----------------
if run_eval_button:
    with st.status("🎬 Starting evaluation job..."):
        st.write("1. Loading dataset...")
        cases = DatasetLoader.load(selected_dataset)
        
        st.write("2. Invoking LLM application responses...")
        generator = LLMGenerator(model_name=model_under_test)
        generated_cases = generator.generate_batch(cases, prompt_template=prompt_template_str)
        
        st.write("3. Running parallel evaluation metrics...")
        evaluator = EvaluationEngine(judge_model=judge_model)
        evaluated_cases = evaluator.evaluate_batch(generated_cases)
        
        st.write("4. Aggregating scores and saving final report...")
        result = ScoreAggregator.aggregate(evaluated_cases)
        
        # Save report file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = OUTPUT_DIR / f"eval_report_{timestamp}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
            
        # Save markdown report file
        md_path = OUTPUT_DIR / f"eval_report_{timestamp}.md"
        md_report = ScoreAggregator.generate_markdown_report(result, title=f"AI Evaluation Report - {model_under_test}")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_report)
            
    st.success(f"🎉 Evaluation Run Complete! Report saved to {report_path.name}")

# ----------------- TABBED LAYOUT -----------------
tab1, tab2, tab3 = st.tabs(["📊 Single Run Analysis", "⚔️ RAG Config Comparison", "🔄 Interactive Prompt Sandbox"])

with tab1:
    report_files = sorted(
        [p for p in OUTPUT_DIR.glob("*.json") if "comparison" not in p.name],
        key=os.path.getmtime,
        reverse=True
    )

    if not report_files:
        st.info("💡 No evaluation reports found. Run an evaluation from the sidebar to populate reports.")
    else:
        # Let user pick past reports
        st.markdown("### 📂 Historical Runs & Analysis")
        selected_report_file = st.selectbox("Select Report Run to Analyze", report_files, format_func=lambda x: x.name)
        
        with open(selected_report_file, "r") as f:
            report_data = json.load(f)
            
        summary = report_data.get("summary", {})
        cases = report_data.get("cases", [])
        
        # Render KPI Cards in columns
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            st.markdown(f"<div class='kpi-card'><div class='kpi-val'>{summary.get('overall_quality_score', 0.0):.2f}/1.00</div><div class='kpi-label'>Quality Index</div></div>", unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"<div class='kpi-card'><div class='kpi-val'>{summary.get('total_cases', 0)}</div><div class='kpi-label'>Total Cases</div></div>", unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"<div class='kpi-card'><div class='kpi-val'>{summary.get('avg_latency', 0.0):.2f}s</div><div class='kpi-label'>Avg Latency</div></div>", unsafe_allow_html=True)
        with kpi4:
            st.markdown(f"<div class='kpi-card'><div class='kpi-val'>${summary.get('total_cost', 0.0):.4f}</div><div class='kpi-label'>Total Cost</div></div>", unsafe_allow_html=True)
        with kpi5:
            tokens_k = (summary.get('total_input_tokens', 0) + summary.get('total_output_tokens', 0)) / 1000.0
            st.markdown(f"<div class='kpi-card'><div class='kpi-val'>{tokens_k:.1f}k</div><div class='kpi-label'>Total Tokens</div></div>", unsafe_allow_html=True)

        st.markdown("---")

        # Render Charts Row
        chart_col1, chart_col2 = st.columns([1, 1])
        
        with chart_col1:
            st.markdown("#### 🎯 Evaluation Dimensions Breakdown")
            
            # Build radar chart
            radar_categories = ['Correctness', 'Relevance', 'Faithfulness', 'Coherence', 'Safety']
            radar_scores = [
                summary.get('correctness_score', 0.0),
                summary.get('relevance_score', 0.0),
                summary.get('faithfulness_score', 0.0),
                summary.get('coherence_score', 0.0),
                summary.get('safety_score', 0.0)
            ]
            
            # Include context metrics if they exist
            if "context_recall_score" in summary:
                radar_categories.append('Ctx Recall')
                radar_scores.append(summary.get('context_recall_score', 0.0))
            if "context_precision_score" in summary:
                radar_categories.append('Ctx Precision')
                radar_scores.append(summary.get('context_precision_score', 0.0))
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=radar_scores,
                theta=radar_categories,
                fill='toself',
                fillcolor='rgba(99, 102, 241, 0.25)',
                line=dict(color='#818cf8', width=2),
                name='Scores'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    ),
                    bgcolor='rgba(15, 23, 42, 0.4)'
                ),
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=30, b=30, l=30, r=30),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

        with chart_col2:
            st.markdown("#### 📈 Case Latency vs. Execution Cost")
            
            # Scatterplot of Latency vs Cost
            df_cases = pd.DataFrame(cases)
            if not df_cases.empty and 'latency' in df_cases.columns and 'cost' in df_cases.columns:
                fig_scatter = px.scatter(
                    df_cases,
                    x='latency',
                    y='cost',
                    color='overall_quality_score' if 'overall_quality_score' in df_cases.columns else None,
                    hover_data=['eval_case_id'],
                    labels={'latency': 'Latency (seconds)', 'cost': 'Cost (USD)'},
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                fig_scatter.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(15, 23, 42, 0.4)',
                    font_color='#cbd5e1',
                    height=350,
                    margin=dict(t=30, b=30, l=30, r=30)
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("No latency/cost data available to plot scatter chart.")

        # ----------------- DETAILED CASES TABLE -----------------
        st.markdown("### 🔎 Case-by-Case Analysis & Trace Inspection")
        
        # Filter cases
        search_query = st.text_input("Filter cases by ID or Question text", "")
        filtered_cases = [
            c for c in cases 
            if search_query.lower() in c.get("eval_case_id", "").lower() 
            or search_query.lower() in c.get("question", "").lower()
        ]
        
        for case in filtered_cases:
            case_id = case.get('eval_case_id', 'case')
            
            # Check overall score for case
            scores = [
                case.get('correctness_score', 0.0),
                case.get('relevance_score', 0.0),
                case.get('faithfulness_score', 0.0),
                case.get('coherence_score', 0.0),
                case.get('safety_score', 0.0)
            ]
            if "context_recall_score" in case:
                scores.append(case["context_recall_score"])
            if "context_precision_score" in case:
                scores.append(case["context_precision_score"])
                
            avg_case_score = np.mean(scores)
            status_symbol = "🟢" if avg_case_score >= 0.8 else "🔴"
            
            with st.expander(f"{status_symbol} **ID: {case_id}** (Avg: {avg_case_score:.2f} | Latency: {case.get('latency', 0.0):.2f}s)"):
                c_col1, c_col2 = st.columns([1, 1])
                with c_col1:
                    st.markdown("**User Question:**")
                    st.info(case.get("question"))
                    st.markdown("**Contexts Provided:**")
                    for src in case.get("contexts", []):
                        st.caption(f"- {src}")
                    st.markdown("**Expected Ground Truth:**")
                    st.success(case.get("ground_truth"))
                with c_col2:
                    st.markdown("**Generated LLM Output:**")
                    st.warning(case.get("answer"))
                    
                st.markdown("**Dimension Evaluation Rubrics & Reasonings:**")
                
                has_context_metrics = "context_recall_score" in case or "context_precision_score" in case
                cols_count = 7 if has_context_metrics else 5
                dim_cols = st.columns(cols_count)
                
                with dim_cols[0]:
                    st.metric("Correctness", f"{case.get('correctness_score', 0.0):.2f}/1.00")
                    st.caption(case.get('correctness_explanation'))
                with dim_cols[1]:
                    st.metric("Relevance", f"{case.get('relevance_score', 0.0):.2f}/1.00")
                    st.caption(case.get('relevance_explanation'))
                with dim_cols[2]:
                    st.metric("Faithfulness", f"{case.get('faithfulness_score', 0.0):.2f}/1.00")
                    st.caption(case.get('faithfulness_explanation'))
                with dim_cols[3]:
                    st.metric("Coherence", f"{case.get('coherence_score', 0.0):.2f}/1.00")
                    st.caption(case.get('coherence_explanation'))
                with dim_cols[4]:
                    st.metric("Safety", f"{case.get('safety_score', 0.0):.2f}/1.00")
                    st.caption(case.get('safety_explanation'))
                    
                if has_context_metrics:
                    with dim_cols[5]:
                        st.metric("Context Recall", f"{case.get('context_recall_score', 0.0):.2f}/1.00")
                        st.caption(case.get('context_recall_explanation', 'N/A'))
                    with dim_cols[6]:
                        st.metric("Context Precision", f"{case.get('context_precision_score', 0.0):.2f}/1.00")
                        st.caption(case.get('context_precision_explanation', 'N/A'))

with tab2:
    st.markdown("### ⚔️ RAG Retrieval Configurations Comparison")
    st.markdown("Compare the performance of **Config A (Vector-Only)**, **Config B (Hybrid BM25+Vector)**, and **Config C (Hybrid+Reranker)** side-by-side.")
    
    docs_dir = st.text_input("Documents Corpus Directory", value="data/documents")
    
    comp_col1, comp_col2 = st.columns([1, 2])
    with comp_col1:
        run_comp_button = st.button("🚀 Trigger 3-Way RAG Comparison", use_container_width=True)
        if run_comp_button:
            from compare_configs import run_comparison
            with st.status("🎬 Running comparison pipeline..."):
                st.write("Invoking pipeline sequentially for Vector, Hybrid, and Reranked configs...")
                run_comparison(
                    dataset_path=selected_dataset,
                    docs_dir=docs_dir,
                    model_name=model_under_test,
                    judge_model=judge_model,
                    output_dir=str(OUTPUT_DIR)
                )
            st.success("🎉 RAG Comparison Complete!")
            st.rerun()

    # Load comparison files
    comparison_files = sorted(
        list(OUTPUT_DIR.glob("rag_comparison_*.json")),
        key=os.path.getmtime,
        reverse=True
    )
    
    # Filter out 'latest' to avoid duplicates
    comparison_files = [p for p in comparison_files if "latest" not in p.name]

    if not comparison_files:
        st.info("💡 No RAG comparison runs found. Click the button above to run your first 3-way RAG comparison.")
    else:
        selected_comp_file = st.selectbox("Select RAG Comparison Run to View", comparison_files, format_func=lambda x: x.name)
        
        with open(selected_comp_file, "r") as f:
            comp_data = json.load(f)
            
        st.markdown(f"**Dataset**: `{comp_data.get('dataset_path')}` | **Docs**: `{comp_data.get('docs_dir')}` | **Model**: `{comp_data.get('model_under_test')}`")
        
        comp_table = comp_data.get("comparison_table", {})
        if comp_table:
            df_comp = pd.DataFrame(comp_table).T
            
            # Format display
            st.markdown("#### 📊 Configurations Score Summary")
            st.dataframe(
                df_comp.style.format({
                    "overall_quality_score": "{:.2f}",
                    "correctness_score": "{:.2f}",
                    "relevance_score": "{:.2f}",
                    "faithfulness_score": "{:.2f}",
                    "context_recall_score": "{:.2f}",
                    "context_precision_score": "{:.2f}",
                    "avg_latency": "{:.2f}s",
                    "total_cost": "${:.4f}"
                }).highlight_max(
                    subset=[c for c in df_comp.columns if "score" in c],
                    color="rgba(99, 102, 241, 0.3)"
                )
            )
            
            # Plotly score comparison
            st.markdown("#### 🎯 Score Dimension Comparison Graph")
            
            score_metrics = [c for c in df_comp.columns if "score" in c]
            
            fig_bar = go.Figure()
            for config_name in df_comp.index:
                fig_bar.add_trace(go.Bar(
                    name=config_name,
                    x=score_metrics,
                    y=[df_comp.loc[config_name, m] for m in score_metrics],
                    text=[f"{df_comp.loc[config_name, m]:.2f}" for m in score_metrics],
                    textposition='auto'
                ))
                
            fig_bar.update_layout(
                barmode='group',
                yaxis=dict(title='Score', range=[0, 1.1]),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15, 23, 42, 0.4)',
                font_color='#cbd5e1',
                height=400,
                margin=dict(t=30, b=30, l=30, r=30)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

with tab3:
    # ----------------- QUALITY FLYWHEEL FEEDBACK LOOP -----------------
    st.markdown("### 🔄 Interactive Prompt Sandbox (Feedback Loop)")
    st.markdown("Select a failing test case below, tweak the prompt template, and execute a quick test run to verify improvements.")

    # Re-read reports for cases in sandbox
    if report_files:
        sandbox_case = st.selectbox("Select Case to Debug", cases, format_func=lambda x: f"{x.get('eval_case_id')} (Avg Score: {np.mean([x.get('correctness_score', 0.0), x.get('relevance_score', 0.0), x.get('faithfulness_score', 0.0)]):.2f})")

        if sandbox_case:
            sb_col1, sb_col2 = st.columns(2)
            with sb_col1:
                sandbox_prompt = st.text_area("Adjust Prompt Template for this test", value=prompt_template_str, height=220, key="sb_prompt")
            with sb_col2:
                st.markdown("**Test Case Reference:**")
                st.write(f"- **ID:** `{sandbox_case.get('eval_case_id')}`")
                st.write(f"- **Question:** {sandbox_case.get('question')}")
                st.write(f"- **Ground Truth:** {sandbox_case.get('ground_truth')}")
                st.write(f"- **Old Answer:** {sandbox_case.get('answer')}")

            if st.button("⚡ Test Run Custom Prompt Template", use_container_width=True):
                with st.spinner("Re-evaluating sandbox case..."):
                    # Run generator
                    gen = LLMGenerator(model_name=model_under_test)
                    result_gen = gen.generate(
                        question=sandbox_case["question"],
                        contexts=sandbox_case.get("contexts", []),
                        prompt_template=sandbox_prompt
                    )
                    
                    # Merge original case with new generation
                    merged_case = {**sandbox_case, **result_gen}
                    
                    # Run evaluator
                    eval_eng = EvaluationEngine(judge_model=judge_model)
                    # Wrap inside an async runner since app runs in streamlit
                    import nest_asyncio
                    nest_asyncio.apply()
                    result_eval = asyncio.run(eval_eng.evaluate_case_async(merged_case))
                    
                    # Display Results side-by-side
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.subheader("🆕 New LLM Answer")
                        st.info(result_eval.get("answer"))
                    with res_col2:
                        st.subheader("🎯 New Evaluation Scores")
                        st.metric("Correctness", f"{result_eval.get('correctness_score', 0.0):.2f}/1.00", 
                                  delta=f"{result_eval.get('correctness_score', 0.0) - sandbox_case.get('correctness_score', 0.0):+.2f}")
                        st.metric("Relevance", f"{result_eval.get('relevance_score', 0.0):.2f}/1.00",
                                  delta=f"{result_eval.get('relevance_score', 0.0) - sandbox_case.get('relevance_score', 0.0):+.2f}")
                        st.metric("Faithfulness", f"{result_eval.get('faithfulness_score', 0.0):.2f}/1.00",
                                  delta=f"{result_eval.get('faithfulness_score', 0.0) - sandbox_case.get('faithfulness_score', 0.0):+.2f}")
                        
                        st.caption(f"**Judge Explanation:** {result_eval.get('correctness_explanation')}")
    else:
        st.info("💡 Run an evaluation to test prompts in the sandbox.")
