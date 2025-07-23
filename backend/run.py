#!/usr/bin/env python3
"""
Simplified startup script for the algorithmic trading system.
This script starts the FastAPI server directly without complex initialization.
"""

import uvicorn
import os
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def main():
    """Start the FastAPI server"""
    print("Starting Algorithmic Trading System...")
    print(f"Backend directory: {backend_dir}")
    
    # Change to backend directory
    os.chdir(backend_dir)
    
    # Start the FastAPI server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()