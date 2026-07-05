import os
from typing import List, Dict, Any
import requests
from openai import OpenAI

class RAGEngine:
    def __init__(self, api_key: str = None, provider: str = "gemini"):
        """
        Initializes the RAG Engine with an LLM provider and its API key.
        Defaults to gemini using the key in environment variables if not provided.
        """
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") if self.provider == "gemini" else os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(f"API key is missing for provider: {self.provider}")
            
        if self.provider == "openai":
            self.client = OpenAI(api_key=self.api_key)
            self.model_name = "gpt-4o-mini"
        elif self.provider != "gemini":
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates a RAG answer for the query based on the context chunks.
        Restricts the LLM from hallucinating answers outside the context.
        Returns a dict: {"answer": answer_text, "citations": list_of_cited_chunks}
        """
        # Format the context text
        formatted_context = ""
        for i, chunk in enumerate(context_chunks):
            formatted_context += f"--- CONTEXT CHUNK {i+1} (Source: {chunk['doc_name']}, Page: {chunk['page']}) ---\n"
            formatted_context += f"{chunk['text']}\n\n"
            
        # Formulate system prompt
        system_instruction = (
            "You are a helpful study assistant. Answer the user's question using ONLY the provided CONTEXT CHUNKS. "
            "For every key statement or fact in your answer, you MUST cite the context chunk number (e.g., [Chunk 1], [Chunk 2]) "
            "that supports it. If the context chunks do not contain the answer, state clearly that you cannot answer based "
            "on the provided document. Do not invent facts."
        )
        
        prompt = f"User Question: {query}\n\n{formatted_context}"
        
        if self.provider == "gemini":
            # Using v1beta endpoint with gemini-2.5-flash as gemini-1.5-flash is not available on this API tier
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],
                "systemInstruction": {
                    "parts": [
                        {"text": system_instruction}
                    ]
                }
            }
            try:
                res = requests.post(url, json=payload, headers=headers)
                res.raise_for_status()
                res_json = res.json()
                answer = res_json["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                error_detail = ""
                if 'res' in locals() and hasattr(res, 'text'):
                    error_detail = f" | Details: {res.text}"
                raise RuntimeError(f"Gemini API request failed: {e}{error_detail}")
        else:  # openai
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.choices[0].message.content
            
        # Extract citations referenced in the text
        citations = []
        for i, chunk in enumerate(context_chunks):
            citation_str = f"Chunk {i+1}"
            if citation_str in answer:
                citations.append({
                    "chunk_num": i + 1,
                    "doc_name": chunk["doc_name"],
                    "page": chunk["page"],
                    "text": chunk["text"]
                })
                
        return {
            "answer": answer,
            "citations": citations
        }
