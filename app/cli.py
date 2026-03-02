import argparse
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules
from src.rag_pipeline import TourismRAG, create_rag_system
from src.evaluation import RAGEvaluator, create_test_file, DEFAULT_TEST_QUESTIONS
from src.config import config

# CLI implementation
def print_banner():
    print("""
🦁 ===========================================
    KENYA TOURISM RAG SYSTEM
    Powered by Grok (xAI) + Local Embeddings
   ===========================================
    """)

# Interactive chat mode
def interactive_mode(rag: TourismRAG):
    print_banner()
    print("Enter your questions about Kenya tourism.")
    print("Commands: 'quit' to exit, 'stats' for system info, 'sources' to toggle sources\n")
    
    show_sources = False
    
    while True:
        try:
            query = input("\n You: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            if query.lower() == 'stats':
                stats = rag.get_stats()
                print(f"\n Stats: {json.dumps(stats, indent=2)}")
                continue
            
            if query.lower() == 'sources':
                show_sources = not show_sources
                print(f"Sources display: {'ON' if show_sources else 'OFF'}")
                continue
            
            # Process query
            print("\n Thinking...")
            result = rag.query(query)
            
            print(f"\n Answer: {result['answer']}")
            
            if show_sources and result.get('retrieved_documents'):
                print(f"\n Sources ({result['retrieved_count']} documents):")
                for i, doc in enumerate(result['retrieved_documents'][:3], 1):
                    meta = doc['metadata']
                    print(f"  {i}. {meta.get('title', 'Unknown')} "
                          f"(Score: {doc.get('score', 0):.3f})")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! ")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

# Main function
def main():
    parser = argparse.ArgumentParser(
        description="Kenya Tourism RAG System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --build                    Build knowledge base
  %(prog)s -q "Best time to visit Maasai Mara?"
  %(prog)s --interactive              Start chat mode
  %(prog)s --eval default             Run evaluation with default questions
  %(prog)s --eval test_questions.json Run evaluation from file
  %(prog)s --eval-init                Create test questions template
  %(prog)s --stats                    Show system statistics
        """
    )
    
    # Define CLI arguments
    parser.add_argument('--build', action='store_true',
                       help='Build knowledge base from CSV files')
    parser.add_argument('-q', '--query', type=str,
                       help='Single query mode')
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='Interactive chat mode')
    parser.add_argument('--eval', type=str, metavar='FILE_OR_DEFAULT',
                       help='Run evaluation ("default" for built-in questions, or path to JSON file)')
    parser.add_argument('--eval-init', action='store_true',
                       help='Create default test questions file')
    parser.add_argument('--stats', action='store_true',
                       help='Show system statistics')
    parser.add_argument('--output', '-o', type=str, default='evaluation/results.json',
                       help='Output file for evaluation results (default: evaluation/results.json)')
    
    args = parser.parse_args()
    
    # Handle eval-init first (no KB needed)
    if args.eval_init:
        filepath = create_test_file()
        print(f" Created test questions template: {filepath}")
        print("Edit this file and run: python -m app.cli --eval evaluation/test_questions.json")
        return
    
    # Validate config
    errors = config.validate()
    if errors:
        print(" Configuration errors:")
        for error in errors:
            print(f"   {error}")
        print("\nPlease set your GROK_API_KEY in .env file or environment")
        sys.exit(1)
    
    # Execute command
    if args.build:
        print(" Building knowledge base...")
        rag = TourismRAG()
        count = rag.build_knowledge_base()
        print(f"\n Built knowledge base with {count} chunks")
        
    elif args.query:
        rag = create_rag_system()
        result = rag.query(args.query)
        print(f"\nQ: {result['query']}\n")
        print(f"A: {result['answer']}\n")
        
        if result.get('sources'):
            print("Sources:")
            for src in result['sources'][:3]:
                print(f"  - {src.get('title')} ({src.get('location')})")
                
    elif args.interactive:
        rag = create_rag_system()
        interactive_mode(rag)
        
    elif args.eval:
        print(f" Running evaluation...")
        
        # Determine questions source
        if args.eval.lower() == 'default':
            questions = DEFAULT_TEST_QUESTIONS
            print(f"Using default {len(questions)} test questions")
        else:
            # Load from file
            try:
                with open(args.eval, 'r') as f:
                    data = json.load(f)
                
                # Handle different formats
                if isinstance(data, list):
                    if len(data) > 0:
                        if isinstance(data[0], str):
                            questions = data
                        elif isinstance(data[0], dict):
                            questions = [item.get('question', item.get('query', '')) for item in data]
                        else:
                            raise ValueError("Unknown question format in file")
                    else:
                        questions = []
                else:
                    raise ValueError("Questions file should contain a list")
                
                print(f"Loaded {len(questions)} test questions from {args.eval}")
                
            except FileNotFoundError:
                print(f" File not found: {args.eval}")
                print("Run 'python -m app.cli --eval-init' to create a template")
                sys.exit(1)
            except json.JSONDecodeError as e:
                print(f" Invalid JSON in {args.eval}: {e}")
                sys.exit(1)
        
        if not questions:
            print(" No questions loaded")
            sys.exit(1)
        
        # Initialize RAG and run evaluation
        rag = create_rag_system()
        evaluator = RAGEvaluator(rag)
        
        # Run evaluation
        metrics = evaluator.run_evaluation(questions, args.output)
        
        # Print report
        print("\n" + evaluator.generate_report())
        
        # Print summary
        print(f"\n Summary:")
        print(f"  Total Questions: {metrics['total_questions']}")
        print(f"  Success Rate: {metrics['success_rate']*100:.1f}%")
        print(f"  Average Response Time: {metrics['avg_response_time']}s")
        print(f"  Average Documents Retrieved: {metrics['avg_documents_retrieved']}")
        print(f"  Failed Queries: {metrics['failed_queries']}")
        print(f"\n Detailed results saved to: {args.output}")
        
    elif args.stats:
        rag = create_rag_system()
        stats = rag.get_stats()
        print(json.dumps(stats, indent=2))
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()