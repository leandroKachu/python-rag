import os
import re, json

from app.ingest import retrieve_context

from dotenv import load_dotenv
from google import genai


load_dotenv()
key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=key)

def ask_llm(question: str) -> str:
    context = retrieve_context(question)
    if not context.strip():
        return "Não encontrei informações suficientes nos documentos."

    prompt = f"""
    Retorne apenas o JSON entre <json></json>

    <json>
    {{
    "answer": "",
    "sources": []
    }}
    </json>

    Contexto:
    {context}

    Pergunta:
    {question}
    Regras:
    - Se a resposta NÃO estiver claramente no contexto, diga exatamente: "Não encontrei essa informação nos documentos."
    - Não invente.
    
    """

    response = client.models.generate_content (
        model="gemini-3-flash-preview",
        contents=prompt,
    )
    

    match = re.search(r"<json>(.*?)</json>", response.text, re.S)

    if match:
        data = json.loads(match.group(1))
    else:
        data = {"response": response.text, "sources": []}
    return data