import sys
import warnings

warnings.warn("Use 'python main.py' instead of 'python run.py'", DeprecationWarning)

if __name__ == "__main__":
    from main import main
    main()