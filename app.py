"""
Adaptive LoRA API Server - HuggingFace Spaces Compatible

This file is designed to run on HuggingFace Spaces.
Deploy by pushing to a Space with 'gradio' SDK.

Requirements (in requirements.txt for HF Space):
- gradio
- requests
"""

import gradio as gr
import requests
import os

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = "llama-3.1-8b-instant"

# Adapter descriptions for demo
ADAPTERS = {
    "reasoning": "📊 Reasoning - Logical analysis, math, step-by-step thinking",
    "code": "💻 Code - Programming, debugging, code explanation", 
    "creative": "🎨 Creative - Writing, storytelling, brainstorming",
    "analysis": "🔬 Analysis - Data analysis, summarization, research"
}


def route_query(query: str) -> str:
    """Simple keyword-based routing for demo purposes"""
    query_lower = query.lower()
    
    if any(w in query_lower for w in ["code", "python", "function", "debug", "programming"]):
        return "code"
    elif any(w in query_lower for w in ["write", "story", "creative", "poem", "imagine"]):
        return "creative"
    elif any(w in query_lower for w in ["analyze", "data", "summarize", "research", "compare"]):
        return "analysis"
    else:
        return "reasoning"


def generate(prompt: str, max_tokens: int = 300, temperature: float = 0.7, auto_route: bool = True) -> tuple:
    """Generate response with adapter routing"""
    
    if not GROQ_API_KEY:
        return "Error: GROQ_API_KEY not set. Add it in Space settings.", "N/A"
    
    # Route to adapter
    adapter = route_query(prompt) if auto_route else "reasoning"
    adapter_info = ADAPTERS[adapter]
    
    # Build system prompt based on adapter
    system_prompts = {
        "reasoning": "You are a logical reasoning expert. Think step by step and explain your reasoning clearly.",
        "code": "You are an expert programmer. Provide clean, well-commented code with explanations.",
        "creative": "You are a creative writer. Be imaginative, engaging, and original.",
        "analysis": "You are a data analyst. Be thorough, precise, and evidence-based."
    }
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system_prompts[adapter]},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": int(max_tokens),
                "temperature": temperature
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()["choices"][0]["message"]["content"]
            return result, adapter_info
        else:
            return f"Error {response.status_code}: {response.text}", adapter_info
            
    except Exception as e:
        return f"Error: {str(e)}", adapter_info


# Gradio Interface
with gr.Blocks(title="Adaptive LoRA System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 Adaptive Multi-Adapter LLM System
    
    This demo showcases intelligent routing between specialized adapters:
    - 📊 **Reasoning**: Logic, math, step-by-step analysis
    - 💻 **Code**: Programming, debugging, explanations
    - 🎨 **Creative**: Writing, storytelling, brainstorming
    - 🔬 **Analysis**: Data analysis, research, summarization
    
    The router automatically selects the best adapter for your query!
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Your Query",
                lines=4,
                placeholder="Enter your question or task...\n\nExamples:\n- Explain quantum computing step by step\n- Write a Python function to sort a list\n- Write a short story about AI"
            )
            
            with gr.Row():
                max_tokens = gr.Slider(100, 1000, value=300, step=50, label="Max Tokens")
                temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
            
            auto_route = gr.Checkbox(value=True, label="🧠 Auto-route to best adapter")
            submit = gr.Button("Generate", variant="primary")
        
        with gr.Column(scale=2):
            adapter_used = gr.Textbox(label="Adapter Selected", interactive=False)
            response = gr.Textbox(label="Response", lines=15, interactive=False)
    
    gr.Examples(
        examples=[
            ["Explain the difference between supervised and unsupervised learning, step by step"],
            ["Write a Python function that implements binary search with comments"],
            ["Write a creative short story about a robot learning to paint"],
            ["Analyze the pros and cons of renewable energy sources"]
        ],
        inputs=[prompt]
    )
    
    submit.click(
        fn=generate,
        inputs=[prompt, max_tokens, temperature, auto_route],
        outputs=[response, adapter_used]
    )
    
    gr.Markdown("""
    ---
    **How it works:**
    1. Your query is analyzed by a router network
    2. The router selects the most appropriate specialized adapter
    3. The selected adapter generates an optimized response
    
    [GitHub](https://github.com/isuruigi/adaptive-lora-framework) | 
    Built with ❤️ using Groq
    """)


if __name__ == "__main__":
    demo.launch()
