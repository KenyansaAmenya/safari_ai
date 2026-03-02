import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from .rag_pipeline import TourismRAG

logger = logging.getLogger(__name__)

# Evaluation module for RAG system performance on Kenya tourism queries.
@dataclass
class EvaluationResult:
    query: str
    answer: str
    retrieved_count: int
    sources: List[Dict]
    response_time: float
    success: bool
    error: Optional[str] = None
    
    # Manual evaluation fields (to be filled by human)
    relevance_score: Optional[int] = None  # 1-5
    correctness_score: Optional[int] = None  # 1-5
    hallucination_detected: Optional[bool] = None
    notes: Optional[str] = None

# RAG Evaluator class to run evaluations on test questions and generate reports.
class RAGEvaluator:

    # Initialize with RAG instance
    def __init__(self, rag: TourismRAG):
        self.rag = rag
        self.results: List[EvaluationResult] = []
    
    # Load test questions from JSON file
    def load_test_questions(self, filepath: str) -> List[str]:
        """Load test questions from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Handle different formats
        if isinstance(data, list):
            if isinstance(data[0], str):
                return data
            elif isinstance(data[0], dict):
                return [item.get('question', item.get('query', '')) for item in data]
        
        raise ValueError("Unsupported test questions format")
    
    # Run evaluation on provided questions and save results
    def run_evaluation(self, 
                      questions: List[str],
                      output_file: Optional[str] = None) -> Dict[str, Any]:

        logger.info(f"Starting evaluation with {len(questions)} questions")
        
        # Reset results for fresh evaluation
        self.results = []
        total_time = 0
        
        # Iterate through questions and evaluate each one
        for i, question in enumerate(questions, 1):
            logger.info(f"Evaluating [{i}/{len(questions)}]: {question[:60]}...")
            
            # Measure response time for each query
            start_time = time.time()
            
            # Query the RAG system and handle potential errors
            try:
                result = self.rag.query(question)
                response_time = time.time() - start_time
                total_time += response_time
                
                # Store evaluation result with all relevant information
                eval_result = EvaluationResult(
                    query=question,
                    answer=result['answer'],
                    retrieved_count=result.get('retrieved_count', 0),
                    sources=result.get('sources', []),
                    response_time=response_time,
                    success=result.get('success', True)
                )

             # Catch and log any exceptions that occur during evaluation   
            except Exception as e:
                logger.error(f"Error evaluating query: {e}")
                eval_result = EvaluationResult(
                    query=question,
                    answer="",
                    retrieved_count=0,
                    sources=[],
                    response_time=time.time() - start_time,
                    success=False,
                    error=str(e)
                )
            
            # Append the evaluation result to the results list
            self.results.append(eval_result)
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Save results
        if output_file:
            self._save_results(output_file)
        
        return metrics
    
    # Calculate evaluation metrics based on results
    def _calculate_metrics(self) -> Dict[str, Any]:
        if not self.results:
            return {}
        
        # Separate successful and failed results for metric calculations
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        avg_response_time = sum(r.response_time for r in successful) / len(successful) if successful else 0
        
        # Retrieval stats
        avg_retrieved = sum(r.retrieved_count for r in successful) / len(successful) if successful else 0
        
        # Compile metrics into a dictionary for reporting
        return {
            'total_questions': len(self.results),
            'successful_queries': len(successful),
            'failed_queries': len(failed),
            'success_rate': len(successful) / len(self.results),
            'avg_response_time': round(avg_response_time, 2),
            'avg_documents_retrieved': round(avg_retrieved, 2),
            'total_time': round(sum(r.response_time for r in self.results), 2),
            'failures': [{'query': r.query, 'error': r.error} for r in failed]
        }
    
    # Save evaluation results to JSON file
    def _save_results(self, filepath: str):
        
        output = {
            'metrics': self._calculate_metrics(),
            'results': [asdict(r) for r in self.results]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    # Generate a human-readable report based on evaluation results
    def generate_report(self) -> str:
        metrics = self._calculate_metrics()
        
        # Build a comprehensive report string with all relevant metrics and details
        report = []
        report.append("=" * 60)
        report.append("KENYA TOURISM RAG - EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        report.append(f"Total Questions: {metrics['total_questions']}")
        report.append(f"Success Rate: {metrics['success_rate']*100:.1f}%")
        report.append(f"Average Response Time: {metrics['avg_response_time']}s")
        report.append(f"Average Documents Retrieved: {metrics['avg_documents_retrieved']}")
        report.append("")
        
        # Highlight any failures for quick identification
        if metrics['failed_queries'] > 0:
            report.append("FAILURES:")
            for failure in metrics['failures']:
                report.append(f"  - {failure['query'][:50]}...")
                report.append(f"    Error: {failure['error']}")
            report.append("")

        # Provide detailed results for each query for in-depth analysis
        report.append("-" * 60)
        report.append("DETAILED RESULTS:")
        report.append("-" * 60)
        
        # Iterate through each evaluation result and include key details in the report
        for i, result in enumerate(self.results, 1):
            report.append(f"\n[{i}] Query: {result.query}")
            report.append(f"    Success: {'✓' if result.success else '✗'}")
            report.append(f"    Response Time: {result.response_time:.2f}s")
            report.append(f"    Documents Retrieved: {result.retrieved_count}")
            if result.answer:
                answer_preview = result.answer[:200].replace('\n', ' ')
                report.append(f"    Answer Preview: {answer_preview}...")
        
        return "\n".join(report)

# Predefined test questions for Kenya tourism
DEFAULT_TEST_QUESTIONS = [
    "Best wildlife destinations in Kenya",
    "What is the best time to visit Maasai Mara?",
    "Budget-friendly hotels near Nairobi National Park",
    "3-day itinerary in Mombasa",
    "Family-friendly activities on the Kenyan coast",
    "How to get to Mount Kenya?",
    "What animals can I see in Amboseli National Park?",
    "Best restaurants in Lamu",
    "Safety tips for safari in Kenya",
    "Cost of hot air balloon ride in Maasai Mara",
    "Visa requirements for Kenya",
    "Best beach resorts in Diani",
    "What to pack for a Kenya safari?",
    "Cultural experiences with Maasai people",
    "Bird watching spots in Kenya"
]

# Helper function to create a test questions file from the predefined list
def create_test_file(output_path: str = "evaluation/test_questions.json"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(DEFAULT_TEST_QUESTIONS, f, indent=2)
    
    print(f"Created test questions file: {output_path}")
    return output_path

# Quick evaluation function to run a subset of questions for rapid testing and debugging
def run_quick_evaluation(rag: TourismRAG, questions: Optional[List[str]] = None):
   
    if questions is None:
        questions = DEFAULT_TEST_QUESTIONS[:5]  # Use first 5 for quick test
    
    evaluator = RAGEvaluator(rag)
    metrics = evaluator.run_evaluation(questions)
    
    print(evaluator.generate_report())
    return metrics