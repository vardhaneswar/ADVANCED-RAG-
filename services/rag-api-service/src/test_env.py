import sys
import os
import google.generativeai as genai

print("Python version:", sys.version)
print("Python path:", sys.path)
print("Current directory:", os.getcwd())
print("Google Generative AI version:", genai.__version__)