from dotenv import load_dotenv
import os,traceback
load_dotenv(r'C:\Users\PNW_checkout\Downloads\advanced-multisource-rag-finance\advanced-multisource-rag-finance\.env')
print('GEMINI_API_KEY present:', bool(os.getenv('GEMINI_API_KEY')))
try:
    import google.generativeai as genai
except Exception as e:
    print('SDK import error:', e)
    raise
# configure from env
api = os.getenv('GEMINI_API_KEY')
if api:
    try:
        # different SDK versions use different configure patterns
        if hasattr(genai, 'configure'):
            genai.configure(api_key=api)
        elif hasattr(genai, 'Client'):
            genai = genai.Client(api_key=api)
        print('Configured SDK')
    except Exception as e:
        print('Error configuring SDK:', e)
        traceback.print_exc()

# attempt multiple listing styles
try:
    if hasattr(genai, 'list_models'):
        models = genai.list_models()
    elif hasattr(genai, 'Model') and hasattr(genai.Model, 'list'):
        models = genai.Model.list()
    elif hasattr(genai, 'models') and hasattr(genai.models, 'list'):
        models = genai.models.list()
    else:
        raise RuntimeError('No list_models API found on genai SDK')
    names = []
    for m in models:
        if isinstance(m, dict):
            names.append(m.get('name'))
        else:
            names.append(getattr(m, 'name', str(m)))
    print('MODELS_COUNT', len(names))
    for n in names[:200]:
        print(n)
except Exception as e:
    print('Error listing models:')
    traceback.print_exc()
