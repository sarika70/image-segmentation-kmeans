#!/usr/bin/env python
"""
Quick Start Script
==================
Run this to start the Image Segmentation application.

Usage:
    python run.py
    python run.py --port 8080
    python run.py --debug
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Start Image Segmentation App')
    parser.add_argument('--port', type=int, default=5000, help='Port to run on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser')
    args = parser.parse_args()
    
    # Create necessary directories
    for d in ['uploads', 'models', 'sample_images', 'mlruns', 'artifacts']:
        os.makedirs(d, exist_ok=True)
    
    print("=" * 60)
    print("  Image Segmentation Application")
    print("=" * 60)
    print(f"\n  Starting server on http://{args.host}:{args.port}")
    print(f"  Debug mode: {args.debug}")
    print("\n  Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    # Open browser if not disabled
    if not args.no_browser and args.host in ['0.0.0.0', '127.0.0.1', 'localhost']:
        import webbrowser
        import threading
        url = f"http://localhost:{args.port}"
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    
    # Run the app
    from app import app
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()
