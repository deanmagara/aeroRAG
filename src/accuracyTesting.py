"""
Accuracy Testing & Profiling
Evaluates RAG performance on specific aerospace domains.
"""

import time
import pandas as pd
from typing import List, Dict, Any
from .complete_rag_system import CompleteRAGSystem
from .domainConfig import DomainParameters

# Golden set of queries to test specific domain knowledge
AEROSPACE_TEST_QUERIES = [
    {
        "category": "Propulsion",
        "query": "What are the advantages of Hall Effect Thrusters?",
        "expected_keywords": ["efficiency", "isp", "specific impulse", "ionization"]
    },
    {
        "category": "Materials",
        "query": "Describe the thermal properties of phenolic impregnated carbon ablators (PICA).",
        "expected_keywords": ["heat shield", "thermal conductivity", "density", "ablation"]
    },
    {
        "category": "Aerodynamics",
        "query": "Explain the concept of boundary layer ingestion.",
        "expected_keywords": ["drag", "propulsor", "wake", "efficiency"]
    }
]

class AccuracyTester:
    def __init__(self, rag_system: CompleteRAGSystem):
        self.rag_system = rag_system
        self.results = []

    def evaluate_response_quality(self, response: str, expected_keywords: List[str]) -> float:
        """Simple heuristic score based on keyword presence."""
        response_lower = response.lower()
        matched = [k for k in expected_keywords if k.lower() in response_lower]
        return len(matched) / len(expected_keywords) if expected_keywords else 0.0

    def run_comprehensive_test(self):
        """Run all test queries and profile performance."""
        print("\n" + "="*60)
        print("🚀 STARTING AEROSPACE ACCURACY & PERFORMANCE TEST")
        print("="*60)
        
        for item in AEROSPACE_TEST_QUERIES:
            query = item['query']
            print(f"\nTesting: {query}")
            
            # 1. Profile Retrieval Time (explicit check)
            start_retrieval = time.time()
            # Use domain parameters for k
            retrieval_res = self.rag_system.retrieval_system.retrieve(
                query, 
                k=DomainParameters.RETRIEVAL_K
            )
            retrieval_time = time.time() - start_retrieval
            
            # 2. Profile Generation Time (full pipeline)
            start_gen = time.time()
            response = self.rag_system.query(
                query, 
                k=DomainParameters.RETRIEVAL_K,
                use_grounding=True
            )
            gen_time = time.time() - start_gen
            
            # 3. Evaluate Accuracy
            answer_text = response['answer']
            accuracy_score = self.evaluate_response_quality(answer_text, item['expected_keywords'])
            
            self.results.append({
                "Category": item['category'],
                "Retrieval Time (s)": round(retrieval_time, 4),
                "Generation Time (s)": round(gen_time, 4),
                "Keyword Match Score": round(accuracy_score, 2),
                "Num Sources": len(response.get('sources', [])),
                "Response Length": len(answer_text)
            })
            
            print(f"   ⏱️  Ret: {retrieval_time:.2f}s | Gen: {gen_time:.2f}s")
            print(f"   🎯 Accuracy Score: {accuracy_score:.2f}")

        # Summary
        df = pd.DataFrame(self.results)
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        print(df.to_string(index=False))
        return df