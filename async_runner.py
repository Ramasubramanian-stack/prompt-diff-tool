import os
import time
import json
import asyncio
from typing import Dict, Any
from datetime import datetime
from openai import AsyncOpenAI

from dotenv import load_dotenv
load_dotenv()

class MultiProviderDiffRunner:
    def __init__(self, max_concurrent_requests: int = 2):
        # Read keys directly from secure environment variables
        groq_key = os.getenv("GROQ_API_KEY")
        nvidia_key = os.getenv("NVIDIA_API_KEY")

        # Validate presence of keys before initiating connections
        if not groq_key:
            raise ValueError("❌ Missing 'GROQ_API_KEY' in your environment configuration.")
        if not nvidia_key:
            raise ValueError("❌ Missing 'NVIDIA_API_KEY' in your environment configuration.")

        # Initialize OpenAI-compatible network endpoints
        self.groq_client = AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key
        )
        self.nvidia_client = AsyncOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_key
        )
        
        # Guardrail structure ensuring we don't trigger 429 rate limit errors
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def _execute_call(self, client: AsyncOpenAI, prompt: str, model: str, temperature: float) -> Dict[str, Any]:
        start_time = time.perf_counter()
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            latency = time.perf_counter() - start_time
            
            return {
                "status": "success",
                "output": response.choices[0].message.content,
                "latency_seconds": round(latency, 3),
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "finish_reason": response.choices[0].finish_reason,
                "error": None
            }
        except Exception as e:
            return {
                "status": "error",
                "output": "",
                "latency_seconds": round(time.perf_counter() - start_time, 3),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "finish_reason": "error",
                "error": str(e)
            }

    async def worker(self, provider: str, prompt: str, model: str, temperature: float, run_id: int) -> Dict[str, Any]:
        """Locks incoming operations to respect rate limitations and runs exponential retries."""
        async with self.semaphore:
            max_retries = 3
            backoff = 4  
            
            client = self.groq_client if provider == "groq" else self.nvidia_client
            
            for attempt in range(max_retries):
                result = await self._execute_call(client, prompt, model, temperature)
                
                if result["status"] == "success":
                    result["run_id"] = run_id
                    return result
                
                # Intercept rate limit signatures gracefully
                err_msg = str(result["error"]).lower()
                if "429" in err_msg or "rate limit" in err_msg:
                    wait_time = backoff * (attempt + 1)
                    print(f"⚠️ [Run {run_id}] Rate limits triggered on {provider}. Backing off for {wait_time}s...")
                    await asyncio.sleep(wait_time)
                elif attempt < max_retries - 1:
                    await asyncio.sleep(1.5)
            
            result["run_id"] = run_id
            return result

    async def run_experiment(
        self, 
        prompt_name: str, 
        prompt_text: str, 
        provider: str, 
        model: str, 
        num_runs: int = 10,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        print(f"🚀 [Target: {provider.upper()}] Launching '{prompt_name}' using {model} ({num_runs} runs)...")
        
        tasks = [
            self.worker(provider, prompt_text, model, temperature, idx + 1)
            for idx in range(num_runs)
        ]
        
        results = await asyncio.gather(*tasks)
        successful_runs = [r for r in results if r["status"] == "success"]
        print(f"✅ Context complete. {len(successful_runs)}/{num_runs} queries verified.")
        
        return {
            "metadata": {
                "prompt_name": prompt_name,
                "prompt_text": prompt_text,
                "provider": provider,
                "model": model,
                "temperature": temperature,
                "total_runs": num_runs,
                "timestamp": datetime.utcnow().isoformat()
            },
            "runs": results
        }

async def main():
    # Set concurrent pipelines down to 2 to maintain safe pipeline passage
    runner = MultiProviderDiffRunner(max_concurrent_requests=2)
    
    test_prompt = """ Explain quantum entanglement to a high schooler.
    Constraint 1: You must use the analogy of two magical coins.
    Constraint 2: Do not use any other analogies.
    Constraint 3: The explanation must be exactly 3 paragraphs long.
    """
    
    # Execution Variant 1: Groq Engine Pipeline
    groq_logs = await runner.run_experiment(
        prompt_name="Quantum_Entanglement_Groq",
        prompt_text=test_prompt,
        provider="groq",
        model="llama-3.1-8b-instant", 
        num_runs=10,
        temperature=0.7
    )
    with open("groq_experiment_logs.json", "w") as f:
        json.dump(groq_logs, f, indent=4)
        
    # Execution Variant 2: NVIDIA NIM Engine Pipeline
    nvidia_logs = await runner.run_experiment(
        prompt_name="Quantum_Entanglement_Nvidia_Constrained",
        prompt_text=test_prompt,
        provider="nvidia",
        model="meta/llama-3.3-70b-instruct", 
        num_runs=10,
        temperature=0.7
    )
    with open("nvidia_experiment_logs_constrained.json", "w") as f:
        json.dump(nvidia_logs, f, indent=4)
        
    print("\n📊 Run execution successful. Review your JSON payload results on disk.")

if __name__ == "__main__":
    asyncio.run(main())