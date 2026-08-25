"""
Top-level evaluation entry point for Aster & Row AI Support Agent.
Run with: python run_evaluations.py
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from evaluation.evaluate import main

if __name__ == "__main__":
    main()
