from typing import Dict, List
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).resolve().parents[4] / ".env"
load_dotenv(dotenv_path=env_path)

# Configuration from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
USE_REAL_LLM = os.getenv("USE_REAL_LLM", "true").lower() == "true"
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "models/gemini-2.5-pro")

# Gemini client placeholders (initialized lazily)
genai = None
MODEL = None
GEMINI_AVAILABLE = False

if USE_REAL_LLM and GEMINI_API_KEY:
    try:
        import google.generativeai as genai_mod
        genai = genai_mod
        genai.configure(api_key=GEMINI_API_KEY)
        # Defer creating a model instance until the first request to avoid import-time model validation
        GEMINI_AVAILABLE = True
        print(f"[llm_client] ✅ Gemini configured (model={GEMINI_MODEL_NAME})")
    except Exception as e:
        print(f"[llm_client] ⚠️ Gemini initialization failed: {e}")
        GEMINI_AVAILABLE = False
else:
    print("[llm_client] ℹ️ Using stub LLM (set USE_REAL_LLM=true and GEMINI_API_KEY in .env to enable)")


def generate_answer(query: str, context: Dict) -> str:
    """Generate answer using Gemini or stub"""
    docs = context.get("documents", [])
    graph = context.get("graph", [])
    global MODEL
    
    # Build context string
    context_lines = []
    for d in docs:
        context_lines.append(f"• [{d.get('source', 'data')}] {d.get('text', '')}")
    for g in graph:
        head = g.get('head', '?')
        rel = g.get('relation', '?')
        tail = g.get('tail', '?')
        context_lines.append(f"• {head} --{rel}--> {tail}")
    
    context_str = "\n".join(context_lines) if context_lines else "(No context)"
    
    # Try Gemini if available
    if GEMINI_AVAILABLE:
        try:
            prompt = f"""You are a financial analyst AI.

Query: {query}

Context:
{context_str}

Provide clear analysis of key differences, risks, and growth."""

            print("[llm_client] 🤖 Calling Gemini...")
            # Attempt multiple SDK usage patterns to be robust across versions
            resp_text = None
            try:
                # Prefer model object if supported
                if MODEL is None and genai is not None and hasattr(genai, 'GenerativeModel'):
                    try:
                        # cache model object to avoid recreating on each call
                        MODEL = genai.GenerativeModel(GEMINI_MODEL_NAME)
                    except Exception:
                        MODEL = None

                if MODEL is not None and hasattr(MODEL, 'generate_content'):
                    # call with simple signature first (some SDK versions don't accept kwargs)
                    response = MODEL.generate_content(prompt)
                    resp_text = getattr(response, 'text', None)

                # Fallback to top-level genai.generate (different SDK shapes)
                if not resp_text and genai is not None and hasattr(genai, 'generate'):
                    try:
                        response2 = genai.generate(model=GEMINI_MODEL_NAME, prompt=prompt, temperature=0.3, max_output_tokens=512)
                        resp_text = getattr(response2, 'text', None)
                        if not resp_text:
                            candidates = getattr(response2, 'candidates', None)
                            if candidates:
                                # try typical candidate shapes
                                first = candidates[0]
                                resp_text = getattr(first, 'content', None) or (first.get('content') if isinstance(first, dict) else None)
                    except Exception:
                        resp_text = None

                if resp_text:
                    print("[llm_client] ✅ Success")
                    return resp_text.strip()
                else:
                    print("[llm_client] ⚠️ Gemini returned no text; falling back to stub")

            except Exception as inner_e:
                print(f"[llm_client] ❌ Gemini call failed: {inner_e}")
                # Try to list available models to help diagnose model name issues
                try:
                    if genai is not None and hasattr(genai, 'list_models'):
                        models = genai.list_models()
                        # convert to list in case the SDK returns a generator
                        models_list = list(models) if models is not None else []
                        # models may be objects or dicts; attempt to print ids/names
                        model_ids = []
                        for m in models_list[:30]:
                            mid = getattr(m, 'id', None) or (m.get('id') if isinstance(m, dict) else None) or str(m)
                            model_ids.append(str(mid))
                        print("[llm_client] Available models (sample):", ", ".join(model_ids))
                except Exception as e2:
                    print(f"[llm_client] Could not list models: {e2}")

        except Exception as e:
            print(f"[llm_client] ❌ Error: {e}")
    
    # Stub answer
    return f"""Query: {query}

Retrieved Context:
{context_str}

(Using stub mode - add GEMINI_API_KEY to enable AI)"""