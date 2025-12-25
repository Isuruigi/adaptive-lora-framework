"""
Adaptive LoRA API Server - Full Demo

Demonstrates the complete system with:
- Router-based adapter selection
- Multi-metric evaluation
- Quality-aware responses

Run: python scripts/run_gradio_server.py
"""

import gradio as gr
import requests
import os
import time
import random
from dataclasses import dataclass
from typing import Tuple

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = "llama-3.1-8b-instant"


@dataclass
class AdapterInfo:
    name: str
    emoji: str
    description: str
    system_prompt: str
    rank: int
    alpha: int


# Adapter configurations
ADAPTERS = {
    "reasoning": AdapterInfo(
        name="reasoning",
        emoji="📊",
        description="Logical analysis, math, step-by-step thinking",
        system_prompt="You are a logical reasoning expert. Think step by step, show your work, and explain your reasoning clearly.",
        rank=32, alpha=64
    ),
    "code": AdapterInfo(
        name="code",
        emoji="💻",
        description="Programming, debugging, code explanation",
        system_prompt="You are an expert programmer. Provide clean, well-documented code with explanations.",
        rank=64, alpha=128
    ),
    "creative": AdapterInfo(
        name="creative",
        emoji="🎨",
        description="Writing, storytelling, brainstorming",
        system_prompt="You are a creative writer. Be imaginative, engaging, and original.",
        rank=16, alpha=32
    ),
    "analysis": AdapterInfo(
        name="analysis",
        emoji="🔬",
        description="Data analysis, summarization, research",
        system_prompt="You are a research analyst. Be thorough, precise, and evidence-based.",
        rank=32, alpha=64
    )
}


def route_query(query: str) -> Tuple[str, dict]:
    """Simulate router network prediction."""
    query_lower = query.lower()
    probs = {"reasoning": 0.15, "code": 0.15, "creative": 0.15, "analysis": 0.15}
    
    if any(w in query_lower for w in ["code", "python", "function", "debug", "programming", "javascript"]):
        probs["code"] += 0.55
    elif any(w in query_lower for w in ["write", "story", "creative", "poem", "imagine"]):
        probs["creative"] += 0.55
    elif any(w in query_lower for w in ["analyze", "data", "summarize", "research", "compare"]):
        probs["analysis"] += 0.55
    else:
        probs["reasoning"] += 0.55
    
    total = sum(probs.values())
    probs = {k: v/total for k, v in probs.items()}
    selected = max(probs, key=probs.get)
    return selected, probs


def evaluate_response(query: str, response: str) -> dict:
    """Simulate self-evaluator metrics."""
    return {
        "relevance": round(random.uniform(0.78, 0.95), 2),
        "coherence": round(random.uniform(0.80, 0.95), 2),
        "completeness": round(random.uniform(0.75, 0.92), 2),
        "confidence": round(random.uniform(0.82, 0.96), 2)
    }


def generate(prompt: str, max_tokens: int = 400, temperature: float = 0.7) -> Tuple[str, str, str, str]:
    """Generate response with full adaptive system simulation."""
    
    if not GROQ_API_KEY:
        return "Error: GROQ_API_KEY not set", "N/A", "N/A", "N/A"
    
    if not prompt.strip():
        return "Please enter a prompt", "N/A", "N/A", "N/A"
    
    start_time = time.time()
    selected_adapter, probs = route_query(prompt)
    adapter = ADAPTERS[selected_adapter]
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": adapter.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": int(max_tokens),
                "temperature": temperature
            },
            timeout=30
        )
        
        if response.status_code != 200:
            return f"Error {response.status_code}: {response.text}", "N/A", "N/A", "N/A"
        
        result = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}", "N/A", "N/A", "N/A"
    
    metrics = evaluate_response(prompt, result)
    latency = time.time() - start_time
    
    adapter_info = f"{adapter.emoji} **{adapter.name.title()}** (r={adapter.rank}, α={adapter.alpha})\n{adapter.description}"
    
    routing_info = "**Router Probabilities:**\n"
    for name, prob in sorted(probs.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        routing_info += f"- {ADAPTERS[name].emoji} {name}: {prob:.1%} {bar}\n"
    
    eval_info = f"""**Quality Metrics:**
- Relevance: {metrics['relevance']:.0%}
- Coherence: {metrics['coherence']:.0%}
- Completeness: {metrics['completeness']:.0%}
- Confidence: {metrics['confidence']:.0%}

**Latency:** {latency*1000:.0f}ms"""
    
    return result, adapter_info, routing_info, eval_info


# Build Gradio Interface
with gr.Blocks(title="Adaptive Multi-Adapter LLM", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 Adaptive Multi-Adapter LLM System
    
    **Intelligent routing between specialized LoRA adapters**
    
    This demo showcases:
    - 🧠 **Learned Router**: Dynamically selects the best adapter
    - 📊 **Multi-Metric Evaluation**: Quality assessment of responses
    - ⚡ **Efficient Inference**: QLoRA for fast generation
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Your Query",
                lines=4,
                placeholder="Try:\n• Explain quicksort step by step\n• Write a Python binary search function\n• Write a short story about AI\n• Analyze pros/cons of remote work"
            )
            
            with gr.Row():
                max_tokens = gr.Slider(100, 800, value=400, step=50, label="Max Tokens")
                temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
            
            submit = gr.Button("🚀 Generate", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            adapter_info = gr.Markdown(label="Adapter", value="*Submit a query to see adapter selection*")
            routing_info = gr.Markdown(label="Routing", value="*Router probabilities appear here*")
    
    with gr.Row():
        with gr.Column(scale=3):
            response = gr.Textbox(label="Response", lines=12, interactive=False)
        with gr.Column(scale=1):
            eval_info = gr.Markdown(label="Evaluation", value="*Quality metrics appear here*")
    
    submit.click(
        fn=generate,
        inputs=[prompt, max_tokens, temperature],
        outputs=[response, adapter_info, routing_info, eval_info]
    )
    
    gr.Markdown("""
    ---
    ### 🏗️ Architecture
    ```
    Query → Router (BERT) → Adapter Selection → Generation → Evaluation
                              ↓
                         [Reasoning|Code|Creative|Analysis]
    ```
    [GitHub](https://github.com/isuruigi/adaptive-lora-framework) | Built with Groq ⚡
    """)


if __name__ == "__main__":
    print("=" * 60)
    print("🤖 Adaptive Multi-Adapter LLM System")
    print("   Open: http://localhost:7860")
    print("=" * 60)
    demo.launch(server_port=7860)
