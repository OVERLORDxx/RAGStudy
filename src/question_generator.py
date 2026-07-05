import json
import os
import re
from typing import List, Dict, Any, Tuple
import requests
from openai import OpenAI
from src.vector_store import LocalVectorStore

class QuestionGenerator:
    def __init__(self, api_key: str = None, provider: str = "gemini"):
        """
        Initializes the Question Generator with an LLM provider and its API key.
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

    def generate_questions(
        self, 
        chunks: List[Dict[str, Any]], 
        num_questions: int = 5,
        difficulty: str = "Medium",
        topics: str = "any"
    ) -> List[Dict[str, Any]]:
        """
        Generates exam-style questions (MCQs, short-answer, long-answer) from document chunks.
        Returns a list of question dicts.
        """
        if not chunks:
            return []
            
        # Combine all chunk text to give LLM full context for question generation
        full_context = ""
        for i, c in enumerate(chunks[:20]):  # Limit context chunks to avoid token limits
            full_context += f"[Chunk {i+1}] (Page: {c['page']}): {c['text']}\n\n"
            
        prompt = f"""
You are an expert exam designer. Generate exactly {num_questions} questions from the following source materials.
Difficulty level: {difficulty}
Target Topics: {topics}

The generated questions should consist of:
- Multiple Choice Questions (MCQs) - include 4 options and the correct answer.
- Short Answer Questions - include the expected answer.
- Long Answer/Essay Questions - include the key points required in the answer.

You MUST format the output as a valid JSON array of objects. Do not write any conversation before or after the JSON.
Each object in the array must follow this exact schema:
{{
    "id": 1,
    "question": "The question text here?",
    "type": "MCQ" or "Short Answer" or "Long Answer",
    "options": ["Option A", "Option B", "Option C", "Option D"], (Leave empty array for Short/Long Answer)
    "answer": "Correct answer text (or option value for MCQ, or key points for Long Answer)",
    "topic": "Specific topic name",
    "difficulty": "{difficulty}"
}}

Source Materials:
{full_context}
"""

        try:
            if self.provider == "gemini":
                # Call Gemini REST API directly with API key in header
                url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
                headers = {
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key
                }
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "systemInstruction": {"parts": [{"text": "You are a professional exam paper generator. Output raw JSON only."}]}
                }
                res = requests.post(url, json=payload, headers=headers)
                res.raise_for_status()
                content = res.json()["candidates"][0]["content"]["parts"][0]["text"]
            else:  # openai
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a professional exam paper generator. Output raw JSON only."},
                        {"role": "user", "content": prompt}
                    ]
                )
                content = response.choices[0].message.content
                
            # Clean up JSON formatting (sometimes LLM wraps it in ```json ... ```)
            cleaned_content = content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]
            cleaned_content = cleaned_content.strip()
            
            questions = json.loads(cleaned_content)
            return questions
        except Exception as e:
            # Fallback mock question if parsing fails
            return [
                {
                    "id": 1,
                    "question": f"Failed to generate questions. Error: {str(e)}",
                    "type": "Short Answer",
                    "options": [],
                    "answer": "Please try again or check your API key / document format.",
                    "topic": "Error",
                    "difficulty": "N/A"
                }
            ]

    def verify_grounding(
        self, 
        question: Dict[str, Any], 
        vector_store: LocalVectorStore,
        top_k: int = 3
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """
        Grounding check: Verifies if a generated question and its answer is supported by the source document.
        Returns:
            - is_grounded: bool
            - explanation: str
            - cited_chunks: list of chunks that verify the question/answer
        """
        q_text = question["question"]
        a_text = str(question["answer"])
        
        # 1. Retrieve the top K chunks for this question
        retrieved = vector_store.search(q_text, top_k=top_k)
        if not retrieved:
            return False, "No source documents found in the database to verify this question.", []
            
        # Format chunks for LLM verification
        verification_context = ""
        for i, (chunk, score) in enumerate(retrieved):
            verification_context += f"[Chunk {i+1}] (Page {chunk['page']}): {chunk['text']}\n\n"
            
        prompt = f"""
You are a facts checker. Verify if the following question and its correct answer are fully supported by, and traceable to, the provided source context chunks.

Question: {q_text}
Expected Answer: {a_text}

Source Context Chunks:
{verification_context}

Provide your assessment in the following format:
VERIFIED: YES or NO
EXPLANATION: A short explanation showing which chunk supports the answer (with page number), or why the answer is not grounded or represents a hallucination.
"""

        try:
            if self.provider == "gemini":
                url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
                headers = {
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key
                }
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "systemInstruction": {"parts": [{"text": "You are a grounding checker. Analyze the statement against context facts."}]}
                }
                res = requests.post(url, json=payload, headers=headers)
                res.raise_for_status()
                res_text = res.json()["candidates"][0]["content"]["parts"][0]["text"]
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a grounding checker. Analyze the statement against context facts."},
                        {"role": "user", "content": prompt}
                    ]
                )
                res_text = response.choices[0].message.content
                
            # Parse response
            is_grounded = False
            explanation = res_text
            
            verified_match = re.search(r"VERIFIED:\s*(YES|NO)", res_text, re.IGNORECASE)
            if verified_match:
                ver_val = verified_match.group(1).upper()
                if ver_val == "YES":
                    is_grounded = True
                    
            explanation_match = re.search(r"EXPLANATION:\s*(.*)", res_text, re.DOTALL | re.IGNORECASE)
            if explanation_match:
                explanation = explanation_match.group(1).strip()
                
            # Filter the cited chunks that were retrieved
            cited_chunks = [item[0] for item in retrieved]
            
            return is_grounded, explanation, cited_chunks
        except Exception as e:
            return False, f"Grounding check failed due to error: {e}", []
