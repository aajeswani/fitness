"""Put the generated module on the path.

fitfood.py lives at the repo root; this suite lives in tests/. Mirrors
../tests/conftest.py in the parent checkout, but scoped to this repo.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
