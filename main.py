import sys
import argparse
import subprocess

# Main entry point for the Kenya Tourism RAG System
def run_cli():
    
    # Run the CLI interface. This will delegate to app.cli.main() which has its own argument parsing.
    from app.cli import main
    sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove 'cli' argument
    main()

# Run the Streamlit web interface
def run_web():
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "app/streamlit_app.py",
        "--server.port=8501",
        "--server.address=localhost"
    ])

# Run the FastAPI server
def run_api():
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)

# Main function to parse arguments and dispatch to the appropriate mode
def main():
    parser = argparse.ArgumentParser(
        description="Kenya Tourism RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py cli --interactive        Start CLI mode
  python main.py cli --build              Build knowledge base
  python main.py web                      Launch web interface
  python main.py api                      Start API server
  python main.py api --port 8080          Start API on custom port
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # CLI command
    cli_parser = subparsers.add_parser('cli', help='Command line interface')
    cli_parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    cli_parser.add_argument('--build', action='store_true', help='Build knowledge base')
    cli_parser.add_argument('-q', '--query', type=str, help='Single query')
    cli_parser.add_argument('--eval', type=str, help='Run evaluation')
    cli_parser.add_argument('--eval-init', action='store_true', help='Create test file')
    
    # Web command
    web_parser = subparsers.add_parser('web', help='Web interface (Streamlit)')
    
    # API command
    api_parser = subparsers.add_parser('api', help='API server (FastAPI)')
    api_parser.add_argument('--host', default='0.0.0.0', help='Host to bind')
    api_parser.add_argument('--port', type=int, default=8000, help='Port to bind')
    api_parser.add_argument('--no-reload', action='store_true', help='Disable auto-reload')
    
    args = parser.parse_args()
    
    if args.command == 'cli':
        # Handle CLI args manually since cli.py has its own parser
        extra_args = []
        if args.interactive:
            extra_args.append('--interactive')
        if args.build:
            extra_args.append('--build')
        if args.query:
            extra_args.extend(['-q', args.query])
        if args.eval:
            extra_args.extend(['--eval', args.eval])
        if args.eval_init:
            extra_args.append('--eval-init')
        
        sys.argv = [sys.argv[0]] + extra_args
        from app.cli import main as cli_main
        cli_main()
        
    elif args.command == 'web':
        run_web()
        
    elif args.command == 'api':
        import uvicorn
        uvicorn.run(
            "src.api:app", 
            host=args.host, 
            port=args.port, 
            reload=not args.no_reload
        )
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()