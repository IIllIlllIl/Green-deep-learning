"""
Functional tests for mutation.py refactoring.
Add project root to Python path to import mutation package.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
