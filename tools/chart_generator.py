"""
Unified Chart Generator
======================
Single source of truth for all chart generation in the agent.
Used by both DATABASE route and CODE_INTERPRETER route.

Big Tech Pattern: DRY (Don't Repeat Yourself)
"""

import pandas as pd
import json
import re
from typing import Dict, Any

def generate_chart_from_data(
    data: pd.DataFrame,
    user_question: str,
    model_provider: str = "openai"
) -> Dict[str, Any]:
    """
    Generate chart configuration using LLM intelligence.
    
    This is the ONLY chart generator in the system.
    Both DATABASE and CODE_INTERPRETER routes use this function.
    
    Args:
        data: DataFrame with chart data
        user_question: User's original question
        model_provider: LLM to use ("openai", "groq", "ollama")
    
    Returns:
        dict: Chart configuration with:
            - chart_type: Type of chart (line, bar, pie, etc)
            - plotly_code: Python code to generate chart
            - reasoning: Why this chart type
            - mode: Chart mode (lines+markers, etc)
    """
    try:
        from adalflow.core import Generator
        from adalflow.components.model_client import OpenAIClient, GroqAPIClient, OllamaClient
        
        # Choose model client
        if model_provider == "groq":
            model_client = GroqAPIClient()
            model_kwargs = {"model": "llama-3.1-8b-instant"}
        elif model_provider == "ollama":
            model_client = OllamaClient()
            model_kwargs = {"model": "qwen2.5:14b"}
        else:
            model_client = OpenAIClient()
            model_kwargs = {"model": "gpt-4o-mini"}
        
        # Prepare data info for LLM
        col1_name = data.columns[0]
        col2_name = data.columns[1]
        sample_values = data.head(5).to_dict('records')
        
        # Create ENHANCED prompt
        prompt = f"""You are a Plotly expert. Generate Python code for the PERFECT chart.

USER REQUEST: {user_question}

DATA STRUCTURE (CRITICAL):
- DataFrame name: df
- Column 1: '{col1_name}' (categories/labels) - type: {data[col1_name].dtype}
- Column 2: '{col2_name}' (values/counts) - type: {data[col2_name].dtype}
- Sample data: {sample_values}
- Total rows: {len(data)}

AVAILABLE CHART TYPES:
1. bar - Compare categories (default choice)
2. line - Show trends over time
3. scatter - Show individual data points
4. pie - Show proportions (max 10 categories)

CODE REQUIREMENTS:
1. Use EXACT column names: df['{col1_name}'] and df['{col2_name}']
2. Create variable 'fig' using plotly.graph_objects (go) or plotly.express (px)
3. For bar/line/scatter: Use x=df['{col1_name}'], y=df['{col2_name}']
4. For pie: Use labels=df['{col1_name}'], values=df['{col2_name}']
5. Add title, axis labels, and hover tooltips
6. Use professional colors and styling
7. NO markdown code blocks - just plain Python code

RESPONSE FORMAT (JSON only):
{{
  "chart_type": "exact_type",
  "plotly_code": "import plotly.graph_objects as go\\nfig = go.Figure(...)\\nfig.update_layout(...)",
  "reasoning": "why this chart type"
}}

Now generate code for: {user_question}"""
        
        # Generate chart config
        generator = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            use_cache=False
        )
        
        result = generator(prompt_kwargs={"input_str": prompt})
        response_text = result.data if hasattr(result, 'data') else str(result)
        
        # Extract JSON
        json_match = re.search(r'\{[^{{}}]*"chart_type"[^{{}}]*"plotly_code"[^{{}}]*\}', response_text, re.DOTALL)
        
        if json_match:
            config = json.loads(json_match.group())
            
            # Validate column usage
            plotly_code = config.get('plotly_code', '')
            if col1_name not in plotly_code or col2_name not in plotly_code:
                print(f"⚠️ LLM code doesn't use correct column names, using fallback")
                raise ValueError("Invalid column usage")
            
            print(f"🎨 Unified chart generator: {config.get('chart_type')} - {config.get('reasoning', '')[:50]}...")
            return config
        else:
            raise ValueError("No valid JSON in LLM response")
            
    except Exception as e:
        print(f"⚠️ Chart generation failed: {e}, using fallback")
        
        # FALLBACK: Simple detection based on data structure and keywords
        columns_lower = [col.lower() for col in data.columns]
        user_q_lower = user_question.lower()
        
        # Multi-series detection
        if len(data.columns) == 3 and 'page_name' in columns_lower:
            return {"chart_type": "line", "mode": "lines+markers", "plotly_code": None}
        
        # Keyword-based fallback
        if 'line' in user_q_lower:
            return {"chart_type": "line", "mode": "lines+markers", "plotly_code": None}
        elif 'pie' in user_q_lower:
            return {"chart_type": "pie", "mode": "none", "plotly_code": None}
        else:
            return {"chart_type": "bar", "mode": "none", "plotly_code": None}


# Export
__all__ = ['generate_chart_from_data']

