#!/usr/bin/env python3
"""
Response Quality Evaluation Framework
======================================

Evaluates generated responses across multiple dimensions:
  1. Retrieval Quality (precision, MRR, contamination, diversity)
  2. Generation Quality (BLEU, ROUGE, token efficiency)
  3. Factual Grounding (hallucination detection)
  4. Tone & Helpfulness (manual grading)

Run: python eval_response_quality.py [--test-set small|medium|large]
"""

import sys
import os
import json
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict

sys.path.append(os.path.dirname(__file__))

# Try importing quality metrics libraries
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import single_meteor_score
    from rouge_score import rouge_scorer
    METRICS_AVAILABLE = True
except ImportError:
    print("⚠️  Install metrics: pip install nltk rouge-score")
    METRICS_AVAILABLE = False


@dataclass
class QualityScore:
    """Aggregated quality metrics for a single response."""
    query: str
    category: str
    
    # Retrieval metrics
    precision_at_k: float
    mrr: float
    contamination: float
    
    # Generation metrics
    bleu_score: float
    rouge_l: float
    meteor_score: float
    
    # Grounding metrics
    hallucination_flag: bool
    grounding_score: float
    
    # Tone & completeness
    tone_score: float
    completeness_score: float
    
    # Efficiency
    tokens_generated: int
    tokens_needed: int
    efficiency: float  # tokens_needed / tokens_generated
    
    # Overall
    overall_score: float


class ResponseQualityEvaluator:
    """Multi-dimensional response quality evaluator."""
    
    def __init__(self):
        """Initialize evaluator with metrics."""
        if METRICS_AVAILABLE:
            self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            self.smooth = SmoothingFunction().method1
        
        # Load reference answers for ground truth comparison
        self.reference_answers = self._load_references()
    
    def _load_references(self) -> Dict[str, str]:
        """Load golden reference answers for test queries."""
        return {
            "My WiFi keeps dropping after a firmware update": (
                "We understand your WiFi is dropping after a firmware update. "
                "Please try these steps: 1) Restart your router and wait 2 minutes. "
                "2) Check the admin panel for firmware rollback option. "
                "3) Verify other devices to isolate if it's device-specific. "
                "If the issue persists, our technical team can schedule a diagnostic call."
            ),
            "I was charged twice for my subscription": (
                "We sincerely apologize for the duplicate charge. "
                "1) Locate both transactions in your bank statement. "
                "2) Provide us your account number and transaction dates. "
                "3) We'll investigate and process a refund within 24-48 hours. "
                "The duplicate charge is usually reversed automatically by your bank within 5-7 days."
            ),
            "How do I return a damaged product": (
                "We're sorry your product arrived damaged. "
                "1) Visit our returns portal with your order number. "
                "2) Select 'damaged on arrival' as the reason. "
                "3) We'll email you a prepaid return label. "
                "4) Ship the item back at no cost. "
                "You'll receive a full refund within 5-7 business days of receipt."
            ),
        }
    
    def evaluate_retrieval(
        self,
        query: str,
        retrieved_docs: List[Dict],
        expected_keywords: List[str],
        expected_category: str,
        top_k: int = 5
    ) -> Tuple[float, float, float, float]:
        """
        Evaluate retrieval quality.
        
        Returns:
            (precision_at_k, mrr, contamination, diversity)
        """
        if not retrieved_docs:
            return 0.0, 0.0, 0.0, 0.0
        
        top_k_docs = retrieved_docs[:top_k]
        
        # 1. Precision @ K: fraction of top-K with expected keywords
        keyword_matches = 0
        for doc in top_k_docs:
            text = (doc.get('answer', '') + ' ' + doc.get('text', '')).lower()
            if any(kw.lower() in text for kw in expected_keywords):
                keyword_matches += 1
        precision_at_k = keyword_matches / len(top_k_docs) if top_k_docs else 0.0
        
        # 2. MRR: Mean Reciprocal Rank (position of first relevant result)
        mrr = 0.0
        for rank, doc in enumerate(top_k_docs, 1):
            text = (doc.get('answer', '') + ' ' + doc.get('text', '')).lower()
            if any(kw.lower() in text for kw in expected_keywords):
                mrr = 1.0 / rank
                break
        
        # 3. Contamination: fraction with cross-category terms
        cross_category_terms = [
            'billing', 'payment', 'refund',
            'technical', 'network', 'outage',
            'return', 'shipping', 'exchange'
        ]
        contaminated = 0
        for doc in top_k_docs:
            text = (doc.get('answer', '') + ' ' + doc.get('text', '')).lower()
            doc_category = doc.get('category', '').lower()
            # Flag if cross-category terms appear but category is different
            if doc_category != expected_category.lower():
                if any(term in text for term in cross_category_terms):
                    contaminated += 1
        contamination = contaminated / len(top_k_docs) if top_k_docs else 0.0
        
        # 4. Diversity: fraction of unique answer texts
        texts = [doc.get('answer', doc.get('text', ''))[:100] for doc in top_k_docs]
        unique_texts = len(set(texts))
        diversity = unique_texts / len(texts) if texts else 0.0
        
        return precision_at_k, mrr, contamination, diversity
    
    def evaluate_generation(
        self,
        generated: str,
        reference: str,
        query: str
    ) -> Tuple[float, float, float, int, int]:
        """
        Evaluate generation quality.
        
        Returns:
            (bleu, rouge_l, meteor, tokens_generated, tokens_needed)
        """
        if not METRICS_AVAILABLE:
            return 0.0, 0.0, 0.0, 0, 0
        
        # Tokenize
        gen_tokens = generated.lower().split()
        ref_tokens = reference.lower().split()
        
        # BLEU-4 score (capped at min of gen/ref length)
        max_n = min(4, len(gen_tokens), len(ref_tokens))
        weights = tuple([1.0/max_n] * max_n) + (0,) * (4 - max_n)
        bleu = sentence_bleu([ref_tokens], gen_tokens, weights=weights, smoothing_function=self.smooth)
        
        # ROUGE-L score
        rouge_scores = self.rouge.score(reference, generated)
        rouge_l = rouge_scores['rougeL'].fmeasure
        
        # METEOR score (requires reference in list format)
        try:
            meteor = single_meteor_score(ref_tokens, gen_tokens)
        except:
            meteor = 0.0
        
        # Token efficiency
        tokens_generated = len(gen_tokens)
        tokens_needed = len(ref_tokens)
        
        return bleu, rouge_l, meteor, tokens_generated, tokens_needed
    
    def detect_hallucinations(
        self,
        generated: str,
        retrieved_docs: List[Dict]
    ) -> Tuple[bool, float]:
        """
        Detect if response contains hallucinated (unsupported) claims.
        
        Returns:
            (has_hallucinations, grounding_score)
        """
        if not retrieved_docs:
            return True, 0.0  # No sources = hallucination risk
        
        # Extract key facts from generated response
        # (Very simplified; in production use NER or LLM-based fact extraction)
        gen_sentences = re.split(r'[.!?]+', generated)
        gen_facts = [s.strip().lower() for s in gen_sentences if len(s.split()) > 3]
        
        # Extract facts from retrieved docs
        doc_text = '\n'.join(
            doc.get('answer', doc.get('text', ''))
            for doc in retrieved_docs[:5]
        ).lower()
        
        # Simple grounding: check if key terms from generation appear in docs
        grounded_facts = 0
        for fact in gen_facts:
            # Extract key terms (nouns, verbs)
            words = fact.split()
            key_words = [w for w in words if len(w) > 4]  # Simple heuristic
            if any(kw in doc_text for kw in key_words):
                grounded_facts += 1
        
        grounding_score = grounded_facts / len(gen_facts) if gen_facts else 1.0
        
        # Flag as hallucination if < 70% grounded
        has_hallucinations = grounding_score < 0.7
        
        return has_hallucinations, grounding_score
    
    def evaluate_tone_and_completeness(
        self,
        generated: str,
        query: str
    ) -> Tuple[float, float]:
        """
        Evaluate tone and completeness heuristically.
        
        Returns:
            (tone_score, completeness_score)
        """
        # Tone: check for professional language
        professional_phrases = [
            'apologize', 'understand', 'help', 'assist',
            'here are', 'the following', 'please',
            'sincerely', 'thank'
        ]
        empathetic_phrases = [
            'understand', 'appreciate', 'sorry',
            'inconvenience', 'frustration'
        ]
        
        gen_lower = generated.lower()
        
        professional_count = sum(
            1 for phrase in professional_phrases
            if phrase in gen_lower
        )
        empathetic_count = sum(
            1 for phrase in empathetic_phrases
            if phrase in gen_lower
        )
        
        # Tone score: 0-1 based on professional + empathetic language
        tone_score = min(
            1.0,
            (professional_count + 0.5 * empathetic_count) / 5
        )
        
        # Completeness: check if response has structure
        has_steps = 'step' in gen_lower or bool(re.search(r'\d+\)', generated))
        has_action = any(verb in gen_lower for verb in ['do', 'try', 'check', 'verify', 'ensure'])
        has_timeline = any(t in gen_lower for t in ['day', 'hour', 'minute', 'week', 'soon'])
        has_escalation = any(e in gen_lower for e in ['escalate', 'contact', 'team', 'support', 'further'])
        
        completeness_score = sum([
            has_steps,
            has_action,
            has_timeline,
            has_escalation
        ]) / 4.0
        
        return tone_score, completeness_score


# Test data
TEST_QUERIES = [
    {
        "query": "My WiFi keeps dropping after a firmware update. What should I do?",
        "category": "Technical",
        "keywords": ["wifi", "firmware", "router", "restart", "update"],
    },
    {
        "query": "I was charged twice for my subscription this month.",
        "category": "Billing and Payments",
        "keywords": ["charge", "duplicate", "refund", "subscription", "billing"],
    },
    {
        "query": "How do I return a damaged product?",
        "category": "Returns and Exchanges",
        "keywords": ["return", "damaged", "refund", "exchange", "label"],
    },
]


def main():
    """Run evaluation on test set."""
    print("\n" + "="*70)
    print("RESPONSE QUALITY EVALUATION")
    print("="*70)
    
    if not METRICS_AVAILABLE:
        print("\n⚠️  Note: BLEU/ROUGE metrics unavailable.")
        print("   Install: pip install nltk rouge-score")
    
    evaluator = ResponseQualityEvaluator()
    
    # Import orchestrator
    try:
        from src.agents.orchestrator import run, get_final_draft
        from src.retrieval.qdrant_store import dense_search
        from src.retrieval.bm25_store import sparse_search
        from src.retrieval.reranker import rerank
    except ImportError as e:
        print(f"\n❌ Failed to import: {e}")
        print("   Make sure your src/ directory is properly set up.")
        return
    
    results = []
    
    for i, test_case in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/{len(TEST_QUERIES)}] {test_case['query'][:50]}...")
        
        query = test_case['query']
        category = test_case['category']
        keywords = test_case['keywords']
        
        # Retrieve documents
        try:
            dense = dense_search(query, category, top_k=20)
            sparse = sparse_search(query, category, top_k=20)
            
            # RRF fusion
            scores = {}
            all_docs = {}
            k = 60
            for rank, doc in enumerate(dense):
                did = doc['id']
                scores[did] = scores.get(did, 0) + 1/(k+rank+1)
                all_docs[did] = doc
            for rank, doc in enumerate(sparse):
                did = doc['id']
                scores[did] = scores.get(did, 0) + 1/(k+rank+1)
                all_docs[did] = doc
            fused = sorted(all_docs.values(), key=lambda d: scores[d['id']], reverse=True)
            
            # Rerank
            reranked = rerank(query, fused, top_k=5)
            
        except Exception as e:
            print(f"   ❌ Retrieval failed: {e}")
            continue
        
        # Evaluate retrieval
        prec, mrr, cont, div = evaluator.evaluate_retrieval(
            query, reranked, keywords, category
        )
        print(f"   Retrieval: P@5={prec:.2f}, MRR={mrr:.2f}, Contamination={cont:.2f}, Diversity={div:.2f}")
        
        # Generate response
        try:
            print(f"   Generating...")
            result = run(query, use_cache=False)
            generated = get_final_draft(result)
            
            if not generated:
                print(f"   ⚠️  No response generated")
                continue
        
        except Exception as e:
            print(f"   ❌ Generation failed: {e}")
            continue
        
        # Get reference answer
        reference = evaluator.reference_answers.get(query, "")
        if not reference:
            print(f"   ⚠️  No reference answer available")
            continue
        
        # Evaluate generation
        bleu, rouge, meteor, tokens_gen, tokens_ref = evaluator.evaluate_generation(
            generated, reference, query
        )
        print(f"   Generation: BLEU={bleu:.3f}, ROUGE-L={rouge:.3f}, METEOR={meteor:.3f}")
        print(f"   Efficiency: {tokens_ref}/{tokens_gen} tokens (ratio: {tokens_ref/tokens_gen:.2f})")
        
        # Evaluate grounding
        halluc, grounding = evaluator.detect_hallucinations(generated, reranked)
        print(f"   Grounding: {grounding:.2f} ({'⚠️  HALLUCINATION RISK' if halluc else '✓ Grounded'})")
        
        # Evaluate tone
        tone, completeness = evaluator.evaluate_tone_and_completeness(generated, query)
        print(f"   Tone: {tone:.2f}, Completeness: {completeness:.2f}")
        
        # Overall score (weighted average)
        overall = (
            0.20 * prec +           # Retrieval precision
            0.15 * mrr +            # First result quality
            0.15 * bleu +           # Semantic overlap with reference
            0.15 * rouge +          # Summary quality
            0.10 * grounding +      # Factual grounding
            0.10 * tone +           # Tone appropriateness
            0.15 * completeness     # Actionability
        )
        print(f"   Overall Score: {overall:.3f}\n")
        
        results.append({
            'query': query,
            'category': category,
            'retrieval': {'precision': prec, 'mrr': mrr, 'contamination': cont, 'diversity': div},
            'generation': {'bleu': bleu, 'rouge': rouge, 'meteor': meteor},
            'grounding': grounding,
            'tone': tone,
            'completeness': completeness,
            'overall': overall
        })
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if results:
        avg_overall = sum(r['overall'] for r in results) / len(results)
        avg_retrieval = sum(r['retrieval']['precision'] for r in results) / len(results)
        avg_generation = sum(r['generation']['bleu'] for r in results) / len(results)
        avg_grounding = sum(r['grounding'] for r in results) / len(results)
        
        print(f"\nAverage Scores:")
        print(f"  Overall      : {avg_overall:.3f}")
        print(f"  Retrieval    : {avg_retrieval:.3f}")
        print(f"  Generation   : {avg_generation:.3f}")
        print(f"  Grounding    : {avg_grounding:.3f}")
        
        # Save results
        with open('data/eval_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to data/eval_results.json")
    else:
        print("\n❌ No results to summarize.")


if __name__ == '__main__':
    main()