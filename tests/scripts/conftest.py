import sys
from pathlib import Path

# Add repo root to sys.path so `import scripts.*` resolves correctly.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
