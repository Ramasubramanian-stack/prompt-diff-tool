import os
import time
import asyncio
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import difflib
from scipy import stats
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

# --- Configuration & Styling ---
st.set_page_config(page_title="Prompt Diff Tool", layout="wide")
load_dotenv()

# --- Async Runner Engine ---
class NvidiaPromptRunner:
    def __init__(self, max_concurrent_requests: int = 2):
        nvidia_key = os.getenv("NVIDIA_API_KEY")
        if not nvidia_key:
            st.error("❌ Missing 'NVIDIA_API_KEY' in your .env file.")
            st.stop()
            
        self.client = AsyncOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_key
        )
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.model = "meta/llama-3.3-70b-instruct"

    async def _execute_call(self, prompt: str) -> dict:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=400  # <-- ADD THIS LINE to cap the output length
            )
            return {"status": "success", "output": response.choices[0].message.content}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _worker(self, prompt: str, attempt_id: int) -> dict:
        async with self.semaphore:
            max_retries = 3
            for attempt in range(max_retries):
                start_time = time.time()
                print(f"🚀 [Run {attempt_id}] Starting attempt {attempt + 1}...")
                
                result = await self._execute_call(prompt)
                
                elapsed = time.time() - start_time
                
                if result["status"] == "success":
                    print(f"✅ [Run {attempt_id}] Success in {elapsed:.2f} seconds.")
                    return result
                
                err_msg = str(result.get("error", "")).lower()
                print(f"⚠️ [Run {attempt_id}] Failed (Attempt {attempt + 1}): {err_msg}")
                
                if "429" in err_msg or "rate limit" in err_msg:
                    wait_time = 4 * (attempt + 1)
                    print(f"⏳ [Run {attempt_id}] Rate limited! Sleeping for {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    await asyncio.sleep(1)
                    
            print(f"❌ [Run {attempt_id}] Completely failed after {max_retries} attempts.")
            return result

    async def run_experiment(self, prompt_text: str, num_runs: int):
        # Update this line inside run_experiment:
        tasks = [self._worker(prompt_text, i) for i in range(num_runs)]
        results = await asyncio.gather(*tasks)
        
        # Filter successful runs
        outputs = [r["output"] for r in results if r["status"] == "success"]
        return outputs

# Wrapper to run async code inside Streamlit
def execute_evaluations(prompt_a, prompt_b, num_runs):
    runner = NvidiaPromptRunner(max_concurrent_requests=3)
    
    async def run_both():
        outputs_a = await runner.run_experiment(prompt_a, num_runs)
        outputs_b = await runner.run_experiment(prompt_b, num_runs)
        return outputs_a, outputs_b
        
    return asyncio.run(run_both())


# --- Math & Analysis Functions ---
@st.cache_resource
def load_encoder():
    return SentenceTransformer('all-MiniLM-L6-v2')

def calculate_metrics(outputs, encoder):
    if len(outputs) < 2:
        return None
        
    lengths = [len(text) for text in outputs]
    embeddings = encoder.encode(outputs)
    sim_matrix = cosine_similarity(embeddings)
    upper_tri = np.triu_indices(len(sim_matrix), k=1)
    mean_sim = np.mean(sim_matrix[upper_tri])
    
    return {
        "outputs": outputs,
        "mean_len": np.mean(lengths),
        "std_dev_len": np.std(lengths),
        "mean_sim": mean_sim,
        "sim_matrix": sim_matrix,
        "lengths": lengths
    }

def display_diff(text1, text2):
    diff = difflib.unified_diff(
        text1.splitlines(), text2.splitlines(), 
        fromfile='Highest Variance Run 1', tofile='Highest Variance Run 2', lineterm=''
    )
    # The main container styling remains the same
    diff_html = "<div style='font-family: monospace; font-size: 14px; background-color: #1e1e1e; padding: 15px; border-radius: 5px; color: #d4d4d4; overflow-x: auto;'>"
    
    for line in diff:
        if line.startswith('+'):
            # CHANGED: Was Green (#4CAF50), is now a modern Soft Blue
            diff_html += f"<span style='color: #3b82f6;'>{line}</span><br>"
        elif line.startswith('-'):
            # CHANGED: Was Red (#F44336), is now a Warm Amber
            diff_html += f"<span style='color: #f59e0b;'>{line}</span><br>"
        elif line.startswith('@@'):
            # Context line (changed to a muted purple for better contrast)
            diff_html += f"<span style='color: #a855f7;'>{line}</span><br>"
        else:
            # Standard unchanged text
            diff_html += f"{line}<br>"
            
    diff_html += "</div>"
    return diff_html


# --- Main UI ---
st.title("Prompt Diff & Variance Engine")
st.markdown("Test two prompt variants against **Llama 3.3 70B** to mathematically get which is more stable.")

encoder = load_encoder()

# --- Input Form ---
with st.form("prompt_form"):
    st.subheader("Configure Experiment")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Variant A")
        prompt_a = st.text_area("Enter your first prompt here:", height=150, value="Explain quantum computing to a beginner.")
    with col2:
        st.markdown("Variant B")
        prompt_b = st.text_area("Enter your heavily constrained prompt here:", height=150, value="Explain quantum computing to a beginner. Constraint 1: Exactly 3 paragraphs. Constraint 2: Use a coin-flipping analogy.")
        
    num_runs = st.slider("Number of Iterations per Prompt (Recommended: 5-20)", min_value=5, max_value=20, value=20, step=2)
    
    submitted = st.form_submit_button("🚀 Run Live Evaluation", use_container_width=True)

# --- Execution & Results ---
if submitted:
    if not prompt_a.strip() or not prompt_b.strip():
        st.warning("Please enter text for both prompts.")
        st.stop()
        
    with st.spinner(f"Please wait... This may take a minute."):
        outputs_a, outputs_b = execute_evaluations(prompt_a, prompt_b, num_runs)
        
    if not outputs_a or not outputs_b:
        st.error("Failed to fetch data from the API. Check your keys or rate limits.")
        st.stop()
        
    st.success(f"✅ Successfully captured {len(outputs_a)} runs for A and {len(outputs_b)} runs for B.")
    
    # Calculate Math
    metrics_a = calculate_metrics(outputs_a, encoder)
    metrics_b = calculate_metrics(outputs_b, encoder)
    
    # --- Display Dashboard ---
    st.divider()
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.subheader("🅰️ Variant A Results (Baseline)")
        c1, c2 = st.columns(2)
        c1.metric("Consistency Score", f"{metrics_a['mean_sim']*100:.1f}/100")
        c2.metric("Length Variance (StdDev)", f"{metrics_a['std_dev_len']:.0f} chars")
        fig_a = px.histogram(pd.DataFrame({"Length": metrics_a['lengths']}), x="Length", title="Length Distribution", nbins=10)
        st.plotly_chart(fig_a, use_container_width=True)
        
        # --- DYNAMIC INSIGHT FOR A ---
        stability_text = "very stable and rarely hallucinates" if metrics_a['mean_sim'] > 0.85 else "unstable with high thematic drift"
        st.info(f"**Data Insight:** This baseline prompt achieved a consistency of **{metrics_a['mean_sim']*100:.1f}**, showing it is {stability_text}. However, a massive variance of **{metrics_a['std_dev_len']:.0f}** proves the model improvised its own length on every run.")

    with res_col2:
        st.subheader("🅱️ Variant B Results (Constrained)")
        c1, c2 = st.columns(2)
        c1.metric("Consistency Score", f"{metrics_b['mean_sim']*100:.1f}/100")
        c2.metric("Length Variance (StdDev)", f"{metrics_b['std_dev_len']:.0f} chars")
        fig_b = px.histogram(pd.DataFrame({"Length": metrics_b['lengths']}), x="Length", title="Length Distribution", nbins=10)
        st.plotly_chart(fig_b, use_container_width=True)
        
        # --- DYNAMIC INSIGHT FOR B ---
        focus_text = "strong topical focus" if metrics_b['mean_sim'] > 0.85 else "moderate topical focus"
        variance_text = f"plummeted to **{metrics_b['std_dev_len']:.0f}**, mathematically proving your rules locked down the output size." if metrics_b['std_dev_len'] < metrics_a['std_dev_len'] else f"stayed high at **{metrics_b['std_dev_len']:.0f}**, showing the model ignored your length constraints."
        
        st.info(f"**Data Insight:** By scoring **{metrics_b['mean_sim']*100:.1f}** in consistency, your strict prompt maintained {focus_text}. Crucially, the variance {variance_text}")

    # --- Statistical Rigor ---
    st.divider()
    st.subheader("📐 Statistical Significance (T-Test)")
    t_stat, p_value = stats.ttest_ind(metrics_a['lengths'], metrics_b['lengths'])
    
    st.markdown(f"**T-Statistic:** {t_stat:.3f} | **P-Value:** {p_value:.4f}")
    if p_value < 0.05:
        st.success("✅ **Statistically Significant:** The difference in response lengths is mathematically proven (p < 0.05).")
    else:
        st.error("❌ **Not Significant:** The difference is too small to mathematically prove a definitive change (p >= 0.05).")

    # --- Diff Inspector ---
    st.divider()
    st.subheader("🔍 Extreme Variance Inspector (Variant A)")
    
    sim_matrix = metrics_a['sim_matrix']
    min_sim_index = np.unravel_index(np.argmin(sim_matrix), sim_matrix.shape)
    run_a, run_b = min_sim_index[0], min_sim_index[1]
    
    # Render the diff
    diff_html = display_diff(outputs_a[run_a], outputs_a[run_b])
    st.markdown(diff_html, unsafe_allow_html=True)
    
    # --- DYNAMIC INSIGHT FOR THE DIFF ---
    st.info(
        "**Data Insight:** This visualizer automatically identifies the two runs that drifted the furthest apart in meaning, calculated using **Cosine Similarity** vector math. "
        "It perfectly illustrates how an unconstrained LLM can explain the exact same topic in entirely different ways across different runs. "
        "The text in **warm amber** shows the concepts generated in the first run, while the text in **soft blue** shows how the model altered or replaced those concepts in the second run."
    )