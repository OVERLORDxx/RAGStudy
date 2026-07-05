import os
import random
import numpy as np
from typing import List, Dict, Any, Tuple
import requests
from openai import OpenAI
from src.vector_store import LocalVectorStore

class RAGEvaluator:
    def __init__(self, api_key: str = None, provider: str = "gemini"):
        """
        Initializes the RAG Evaluator with LLM credentials.
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

    def generate_evaluation_set(
        self, 
        chunks: List[Dict[str, Any]], 
        num_eval_pairs: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generates a synthetic evaluation set: list of {'question': q, 'ground_truth_chunk_id': id}
        Uses the LLM to generate questions based on specific chunks.
        """
        if len(chunks) < num_eval_pairs:
            selected_chunks = chunks
        else:
            selected_chunks = random.sample(chunks, num_eval_pairs)
            
        eval_set = []
        
        for chunk in selected_chunks:
            chunk_text = chunk["text"]
            chunk_id = chunk["chunk_id"]
            
            prompt = f"""
Based ONLY on the following text chunk, generate ONE clear, specific study question that can be answered using this text.
The question should be direct and ask about a key fact in the text.
Do not output any conversation, only output the question.

Text chunk:
"{chunk_text}"
"""
            try:
                if self.provider == "gemini":
                    # Using v1beta endpoint with gemini-2.5-flash as gemini-1.5-flash is not available on this API tier
                    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
                    headers = {
                        "Content-Type": "application/json",
                        "x-goog-api-key": self.api_key
                    }
                    payload = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "systemInstruction": {"parts": [{"text": "You are a teacher creating exam questions. Return the question only."}]}
                    }
                    res = requests.post(url, json=payload, headers=headers)
                    res.raise_for_status()
                    question = res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a teacher creating exam questions. Return the question only."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    question = response.choices[0].message.content.strip()
                
                # Strip quotes if the LLM wrapped the question in them
                if question.startswith('"') and question.endswith('"'):
                    question = question[1:-1]
                if question.strip():
                    eval_set.append({
                        "question": question.strip(),
                        "ground_truth_chunk_id": chunk_id,
                        "ground_truth_text": chunk_text,
                        "page": chunk["page"]
                    })
            except Exception as e:
                # If generation fails for a chunk, skip it
                print(f"Skipping evaluation generation for chunk {chunk_id}: {e}")
                continue
                
        return eval_set

    def evaluate_retrieval(
        self, 
        eval_set: List[Dict[str, Any]], 
        vector_store: LocalVectorStore,
        top_k: int = 4
    ) -> Dict[str, Any]:
        """
        Runs retrieval evaluation.
        Calculates:
          - Hit Rate @ K (was the correct chunk retrieved?)
          - Mean Reciprocal Rank (MRR) @ K (average score based on position)
        """
        if not eval_set:
            return {"hit_rate": 0.0, "mrr": 0.0, "details": []}
            
        hits = 0
        reciprocal_ranks = []
        details = []
        
        for pair in eval_set:
            question = pair["question"]
            gt_id = pair["ground_truth_chunk_id"]
            
            # Retrieve top_k chunks
            retrieved = vector_store.search(question, top_k=top_k)
            retrieved_ids = [chunk["chunk_id"] for chunk, score in retrieved]
            
            # Calculate metrics
            hit = 0
            rr = 0.0
            
            if gt_id in retrieved_ids:
                hit = 1
                hits += 1
                rank = retrieved_ids.index(gt_id) + 1  # 1-indexed rank
                rr = 1.0 / rank
                
            reciprocal_ranks.append(rr)
            
            # Save detail
            details.append({
                "question": question,
                "ground_truth_chunk_id": gt_id,
                "ground_truth_page": pair["page"],
                "retrieved_ranks": retrieved_ids,
                "hit": hit,
                "rr": rr,
                "top_retrieved_text": retrieved[0][0]["text"] if retrieved else "None"
            })
            
        return {
            "hit_rate": float(hits / len(eval_set)),
            "mrr": float(np.mean(reciprocal_ranks)),
            "details": details
        }
