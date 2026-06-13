# import os
# from dotenv import load_dotenv
# from google import genai

# load_dotenv()

# client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# for model in client.models.list():
#     print(model.name)
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

print("Testing...")
vec = embeddings.embed_query("Hello world")
print(len(vec))