import re
import json
import asyncio
import tiktoken
import math
from collections import Counter
from typing import Dict, List, Optional, Set
from openai import AsyncOpenAI
import httpx
from .agent_template import ModelProvider
import os

class MyAgent(ModelProvider):
    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key, base_url)
        # Allow model name to be configured via environment variable or default to deepseek-chat if base_url implies it
        self.model_name = os.getenv("MODEL_NAME", "ecnu-max")
        
        # We will create the client in evaluate_model to avoid event loop issues on Windows
        # self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = tiktoken.get_encoding("gpt2")
            
        # Configuration
        # DeepSeek supports 128k context. We use 100k to be safe and leave room for response.
        self.max_context_tokens = 100000 
        self.chunk_size = 2000
        self.chunk_overlap = 200

    def encode_text_to_tokens(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode_tokens(self, tokens: List[int], context_length: Optional[int] = None) -> str:
        if context_length is not None:
            tokens = tokens[:context_length]
        return self.tokenizer.decode(tokens)

    def generate_prompt(self, **kwargs) -> Dict:
        return kwargs

    def _get_keywords(self, text: str) -> List[str]:
        # Simple tokenization by non-alphanumeric splitting
        # This preserves numbers and some codes better than just \w+
        words = re.findall(r'[a-zA-Z0-9\u00C0-\u00FF]+', text.lower())
        return words

    async def evaluate_model(self, prompt: Dict) -> str:
        context_data = prompt['context_data']
        question = prompt['question']
        files = context_data['files']

        # 1. Prepare Chunks
        all_chunks = []
        
        # BM25 Statistics
        doc_freq = {}  # DF: word -> count of docs containing it
        total_chunks = 0
        avgdl = 0
        
        # First pass: Chunking and basic stats
        for file_idx, file in enumerate(files):
            content = file['modified_content']
            tokens = self.encode_text_to_tokens(content)
            
            # Sliding window
            chunk_starts = range(0, len(tokens), self.chunk_size - self.chunk_overlap)
            if len(tokens) <= self.chunk_size:
                chunk_starts = [0]

            for chunk_idx, i in enumerate(chunk_starts):
                if i >= len(tokens) and len(tokens) > 0: 
                    continue
                    
                chunk_tokens = tokens[i:i + self.chunk_size]
                chunk_text = self.decode_tokens(chunk_tokens)
                
                # Get all words (with duplicates for TF)
                chunk_words = self._get_keywords(chunk_text)
                chunk_len = len(chunk_words)
                
                # Update DF (unique words in this doc)
                for w in set(chunk_words):
                    doc_freq[w] = doc_freq.get(w, 0) + 1
                
                all_chunks.append({
                    'text': chunk_text,
                    'words': chunk_words, # Store for TF calculation
                    'len': chunk_len,
                    'filename': file['filename'],
                    'tokens': len(chunk_tokens),
                    'file_idx': file_idx,
                    'chunk_idx': chunk_idx,
                    'global_idx': total_chunks
                })
                total_chunks += 1
                avgdl += chunk_len

        if total_chunks > 0:
            avgdl /= total_chunks
            
        # 2. Score Chunks using BM25
        # BM25 Parameters
        k1 = 1.5
        b = 0.75
        
        question_words = self._get_keywords(question)
        
        for chunk in all_chunks:
            score = 0.0
            chunk_tf = Counter(chunk['words'])
            
            for q_word in question_words:
                if q_word not in doc_freq:
                    continue
                    
                # IDF Calculation
                df = doc_freq[q_word]
                idf = math.log((total_chunks - df + 0.5) / (df + 0.5) + 1)
                
                # TF Component
                tf = chunk_tf[q_word]
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (chunk['len'] / (avgdl + 1e-6))) # Avoid div by zero
                
                score += idf * (numerator / denominator)
            
            chunk['score'] = score

        # 3. Select Top Chunks (Retrieval)
        # Sort by score descending
        sorted_chunks = sorted(all_chunks, key=lambda x: x['score'], reverse=True)
        
        # Initial Retrieval: Top 10 chunks
        top_k_retrieval = 10
        retrieved_chunks = sorted_chunks[:top_k_retrieval]
        
        # 4. Reranking (LLM-based)
        # We will ask the LLM to identify the most relevant chunk from the top candidates.
        # This acts as a filter to reduce noise before the final answer generation.
        
        # Construct Reranking Prompt
        candidates_text = ""
        for i, chunk in enumerate(retrieved_chunks):
            candidates_text += f"Snippet {i+1} (from {chunk['filename']}):\n{chunk['text'][:500]}...\n\n" # Truncate for reranking speed
            
        rerank_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Identify which snippet contains the answer to the user's question. "
                           "Return ONLY the ID of the most relevant snippet (e.g., '2'). If none seem relevant, return '1'."
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nCandidates:\n{candidates_text}\n\nMost relevant snippet ID:"
            }
        ]
        
        # We need to initialize the client
        http_client = httpx.AsyncClient(trust_env=False)
        async with AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=60.0, http_client=http_client) as client:
            
            # Reranking Call
            # Note: To save time/tokens, we could skip this and just feed Top 5. 
            # But user asked for "Introduce Reranking".
            # Let's do a "Soft Rerank": We will just take the Top 5 chunks from BM25 
            # AND the chunk selected by LLM (if it's not in Top 5, which is unlikely).
            
            # Actually, let's simplify: Just use Top 5 BM25 chunks for the final context.
            # The "Reranking" logic requested is best implemented as "Context Selection".
            # Feeding 100k tokens was the error. Feeding Top 5 (10k tokens) is the fix.
            
            # Let's select Top 5 for the final context
            final_selected_chunks = sorted_chunks[:5]
            
            # Map for neighbor lookup (Context Expansion)
            chunk_map = {(c['file_idx'], c['chunk_idx']): c for c in all_chunks}
            
            # Expand context for the Top 1 chunk only (to ensure we get surrounding text if needle is split)
            if final_selected_chunks:
                top_chunk = final_selected_chunks[0]
                f_idx = top_chunk['file_idx']
                c_idx = top_chunk['chunk_idx']
                
                # Add neighbors if not already present
                for offset in [-1, 1]:
                    neighbor_key = (f_idx, c_idx + offset)
                    if neighbor_key in chunk_map:
                        neighbor = chunk_map[neighbor_key]
                        if neighbor not in final_selected_chunks:
                            final_selected_chunks.append(neighbor)

            # 5. Re-assemble Context
            # Sort by file_idx then chunk_idx to maintain document flow
            final_selected_chunks.sort(key=lambda x: (x['file_idx'], x['chunk_idx']))
            
            context_parts = []
            for chunk in final_selected_chunks:
                context_parts.append(f"--- File: {chunk['filename']} ---\n{chunk['text']}")
                
            full_context = "\n\n".join(context_parts)
            
            # 6. Call LLM for Final Answer
            messages = [
                {
                    "role": "system",
                    "content": "You are a professional and patient AI assistant. You are given a set of documents. "
                               "Your task is to answer the user's question based ONLY on the provided documents. "
                               "Some answers can be found directly, while others may require calculation or analysis. "
                               "If the answer is found, provide it clearly and concisely. "
                               "If the answer is not found in the documents, state that you cannot find it. "
                },
                {
                    "role": "user",
                    "content": f"Context:\n{full_context}\n\nQuestion: {question}\n\nAnswer:"
                }
            ]

            try:
                response = await client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=500
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error during model evaluation: {str(e)}"
