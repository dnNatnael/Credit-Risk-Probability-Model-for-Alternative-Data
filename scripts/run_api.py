#!/usr/bin/env python3
"""
Simple script to run the FastAPI application locally.

Usage:
    python scripts/run_api.py
"""

import uvicorn
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

