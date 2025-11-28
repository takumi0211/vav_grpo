"""Core package for the humid simulator project."""
from __future__ import annotations

from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
PHYVAC_DIR = PROJECT_ROOT / "phyvac"

if PHYVAC_DIR.is_dir():
    phyvac_path = str(PHYVAC_DIR)
    if phyvac_path not in sys.path:
        sys.path.insert(0, phyvac_path)

__all__ = ["PACKAGE_ROOT", "PROJECT_ROOT", "PHYVAC_DIR"]
