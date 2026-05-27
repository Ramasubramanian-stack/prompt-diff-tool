import json
import numpy as np
import difflib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class PromptAnalyzer:
    def __init__(self, log_file: str):
        self.log_file = log_file
        print("⏳ Loading local embedding model (all-MiniLM-L6-v2)...")
        # We use a lightweight, fast local model to generate text vectors
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def load_data(self):
        with open(self.log_file, 'r') as f:
            data = json.load(f)
        # Extract only successful outputs
        self.outputs = [run['output'] for run in data['runs'] if run['status'] == 'success']
        self.metadata = data['metadata']
        print(f"✅ Loaded {len(self.outputs)} successful runs for prompt: {self.metadata['prompt_name']}")

    def calculate_variance(self):
        # Calculate standard deviation of character length
        lengths = [len(text) for text in self.outputs]
        mean_len = np.mean(lengths)
        std_dev = np.std(lengths)
        
        print("\n📊 --- Variance Dashboard ---")
        print(f"Average Length: {mean_len:.1f} characters")
        print(f"Length Variance (Std Dev): {std_dev:.1f} characters")
        print(f"Min Length: {min(lengths)} | Max Length: {max(lengths)}")

    def calculate_consistency(self):
        # Convert all text outputs into mathematical vectors
        embeddings = self.encoder.encode(self.outputs)
        
        # Calculate cosine similarity between all pairs of runs
        sim_matrix = cosine_similarity(embeddings)
        
        # Extract the upper triangle to avoid comparing a run to itself (which is always 1.0)
        upper_tri_indices = np.triu_indices(len(sim_matrix), k=1)
        pairwise_similarities = sim_matrix[upper_tri_indices]
        
        mean_sim = np.mean(pairwise_similarities)
        
        print("\n🧠 --- Semantic Consistency Score ---")
        print(f"Average Similarity Score: {mean_sim * 100:.2f}/100")
        if mean_sim > 0.95:
            print("Verdict: Highly Deterministic 🟢")
        elif mean_sim > 0.85:
            print("Verdict: Stable, but shows variation 🟡")
        else:
            print("Verdict: High Variance / Unstable Prompt 🔴")
            
        return sim_matrix

    def show_diff_extremes(self, sim_matrix):
        # Find the two runs that are the LEAST similar to each other
        min_sim_index = np.unravel_index(np.argmin(sim_matrix), sim_matrix.shape)
        run_a, run_b = min_sim_index[0], min_sim_index[1]
        
        print(f"\n🔍 --- Text Diff (Most Divergent Runs: Run {run_a + 1} vs Run {run_b + 1}) ---")
        
        text1 = self.outputs[run_a].splitlines()
        text2 = self.outputs[run_b].splitlines()
        
        diff = difflib.unified_diff(
            text1, text2, 
            fromfile=f'Run {run_a + 1}', 
            tofile=f'Run {run_b + 1}', 
            lineterm=''
        )
        
        for line in diff:
            if line.startswith('+'):
                print(f"\033[92m{line}\033[0m") # Green for additions
            elif line.startswith('-'):
                print(f"\033[91m{line}\033[0m") # Red for removals
            elif line.startswith('@@'):
                print(f"\033[96m{line}\033[0m") # Cyan for context markers

if __name__ == "__main__":
    # Ensure nvidia_experiment_logs.json is in the same directory
    analyzer = PromptAnalyzer('nvidia_experiment_logs.json')
    analyzer.load_data()
    analyzer.calculate_variance()
    sim_matrix = analyzer.calculate_consistency()
    analyzer.show_diff_extremes(sim_matrix)