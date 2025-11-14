#!/usr/bin/env python3
"""
Simple HTTP server for Loom TypeScript examples
Serves both dist/ and example/ directories
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
from pathlib import Path

class LoomHTTPRequestHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        """Translate URL path to filesystem path"""
        # Remove query string
        path = path.split('?', 1)[0]
        path = path.split('#', 1)[0]
        
        # Get the script directory (typescript/)
        base_dir = Path(__file__).parent
        
        # If path starts with /dist/, serve from dist/
        if path.startswith('/dist/'):
            file_path = base_dir / 'dist' / path[6:]
            return str(file_path)
        
        # Otherwise serve from example/
        if path == '/' or path == '':
            file_path = base_dir / 'example' / 'index.html'
        else:
            file_path = base_dir / 'example' / path.lstrip('/')
        
        return str(file_path)
    
    def end_headers(self):
        """Add CORS headers"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        # Set correct MIME type for WASM
        if self.path.endswith('.wasm'):
            self.send_header('Content-Type', 'application/wasm')
        if self.path.endswith('.js'):
            self.send_header('Content-Type', 'application/javascript')
        super().end_headers()

def run_server(port=8081):
    server_address = ('', port)
    httpd = HTTPServer(server_address, LoomHTTPRequestHandler)
    
    print(f"🚀 Loom TypeScript Example Server")
    print(f"📂 Serving:")
    print(f"   /          → example/index.html")
    print(f"   /dist/*    → dist/")
    print(f"   /example/* → example/")
    print(f"\n🌐 Open: http://localhost:{port}")
    print(f"Press Ctrl+C to stop\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")
        httpd.shutdown()

if __name__ == '__main__':
    run_server()
