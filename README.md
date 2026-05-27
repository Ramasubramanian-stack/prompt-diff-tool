# Prompt Diff Engine ⚡
A deterministic evaluation framework for Large Language Models. Stop guessing if your prompt engineering works—prove it with math.


## 🌐 Live Demo
https://prompt-diff-tool-vbp9bpgywmxutagtj4cmxl.streamlit.app/

## 🎯 The Problem
In enterprise AI development, prompt engineering often relies on "vibes" or single-shot testing. An engineer tweaks a prompt, runs it once, and assumes it works. In production, this leads to edge-case hallucinations, unpredictable formatting, and massive semantic drift. Engineers need a way to mathematically prove that adding a constraint (e.g., "be concise," "output JSON") actually controls the LLM's behavior across multiple independent runs.

## 💡 The Solution
The Prompt Diff Engine is an asynchronous batch-testing pipeline that runs a Baseline prompt and a Constrained prompt through Llama 3.3 70B multiple times. It replaces guesswork with statistical rigor by analyzing the outputs using vector embeddings, calculating semantic consistency, and proving behavioral shifts using a paired T-Test.

## Key Engineering Features

### Asynchronous LLM Orchestration
Built a concurrent API pipeline using asyncio to batch-process requests to NVIDIA NIM, complete with automatic exponential backoff and rate-limit handling (429 errors).

### Semantic Variance Inspector (Worst-Case Detection)
Instead of manually reading 20 outputs, the engine uses sentence-transformers (MiniLM) to calculate the Cosine Similarity between all runs. It automatically isolates and displays the two runs that drifted the furthest apart in meaning, instantly exposing the model's worst hallucinations.

### Statistical Rigor
Calculates the Standard Deviation of output lengths and runs a T-Test (SciPy) to mathematically verify if prompt constraints produced a statistically significant difference in model behavior ($p < 0.05$).

## 🏗️ Architecture & Methodology

### Execution Layer
Takes Variant A and Variant B and fires asynchronous batch requests to NVIDIA's Llama-3.3-70B-Instruct model.

### Embedding Layer
Converts all generated text outputs into high-dimensional vector embeddings using a local HuggingFace sentence transformer.

### Analytics Layer
- Computes an $N \times N$ Cosine Similarity matrix to determine the "Consistency Score" (0-100).
- Calculates length variance ($\sigma$).
- Executes a T-Test on the distribution lengths.

### Presentation Layer
A reactive Streamlit dashboard providing dynamic data insights, Plotly histograms, and a unified markdown diff-viewer.

## 🛠️ Tech Stack

### Frontend/UI
- Streamlit
- Plotly (Interactive Data Viz)

### Backend Pipeline
- Python
- asyncio
- Aiohttp (via OpenAI async client)

### AI & NLP
- NVIDIA NIM (Llama 3.3)
- sentence-transformers
- scikit-learn (Cosine Similarity)

### Math & Stats
- numpy
- pandas
- scipy (Hypothesis Testing)

## 🚀 Quick Start (Local Development)

### Prerequisites
- Python 3.9+
- An NVIDIA Build API Key

### Installation

Clone the repository:
```bash
git clone https://github.com/Ramasubramanian-stack/prompt-diff-tool.git
cd prompt-diff-tool
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Set up environment variables

Create a `.env` file in the root directory and add your NVIDIA API key:
```env
NVIDIA_API_KEY="nvapi-your-key-here"
```

### Run the engine
```bash
streamlit run app.py
```

## Conclusion
The Prompt Diff Engine transitions prompt engineering from intuitive guesswork into a rigorous, quantitative discipline. By integrating high-dimensional vector embeddings with automated statistical testing, this framework provides the definitive proof required for enterprise-grade LLM deployment. It empowers AI engineers to mathematically validate structural constraints, isolate silent semantic drift, and ensure deterministic model behavior in production environments. Ultimately, it replaces the fragility of single-shot testing with the reliability of data science.
