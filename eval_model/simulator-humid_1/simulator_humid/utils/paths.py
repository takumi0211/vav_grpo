"""Centralized filesystem paths for the simulator project."""
from __future__ import annotations

from pathlib import Path

from simulator_humid import PACKAGE_ROOT, PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"
WEATHER_DATA_DIR = PROJECT_ROOT / "weather_data"
PEOPLE_DATA_DIR = PROJECT_ROOT / "people_data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
BASELINE_OUTPUT_DIR = OUTPUTS_DIR / "baseline"
LLM_OUTPUT_DIR = OUTPUTS_DIR / "llm"
RL_OUTPUT_DIR = OUTPUTS_DIR / "rl"
REFERENCE_OUTPUT_DIR = OUTPUTS_DIR / "reference"
FIGURES_OUTPUT_DIR = OUTPUTS_DIR / "figures"
SIMULATION_OUTPUT_DIR = OUTPUTS_DIR / "simulations"
ARCHIVE_DIR = DATA_DIR / "archive"

# Ensure commonly used directories exist when module is imported.
for path in (
    OUTPUTS_DIR,
    BASELINE_OUTPUT_DIR,
    LLM_OUTPUT_DIR,
    RL_OUTPUT_DIR,
    REFERENCE_OUTPUT_DIR,
    FIGURES_OUTPUT_DIR,
    SIMULATION_OUTPUT_DIR,
    ARCHIVE_DIR,
):
    path.mkdir(parents=True, exist_ok=True)

__all__ = [
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
    "DATA_DIR",
    "WEATHER_DATA_DIR",
    "PEOPLE_DATA_DIR",
    "OUTPUTS_DIR",
    "BASELINE_OUTPUT_DIR",
    "LLM_OUTPUT_DIR",
    "RL_OUTPUT_DIR",
    "REFERENCE_OUTPUT_DIR",
    "FIGURES_OUTPUT_DIR",
    "SIMULATION_OUTPUT_DIR",
    "ARCHIVE_DIR",
]
