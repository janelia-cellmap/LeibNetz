"""
HTTP server to serve the LeibNetz network builder UI.

This module provides a simple HTTP server that serves the browser-based
network builder interface for constructing LeibNetz networks visually.
"""

import http.server
from pathlib import Path
import socketserver
import webbrowser


def serve_ui(port: int = 8080, open_browser: bool = True) -> None:
    """
    Start an HTTP server to serve the network builder UI.

    Args:
        port: Port number to serve on (default: 8080)
        open_browser: Whether to automatically open the browser (default: True)
    """
    # Get the directory containing this file
    ui_dir = Path(__file__).parent

    # Create a custom handler to set MIME types correctly
    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            # Set the directory to serve files from
            super().__init__(*args, directory=str(ui_dir), **kwargs)

        def end_headers(self):
            # Add CORS headers to allow local development
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            super().end_headers()

        def guess_type(self, path):
            # Ensure correct MIME types
            mimetype = super().guess_type(path)
            if path.endswith(".js"):
                return "application/javascript"
            return mimetype

    # Create the server
    with socketserver.TCPServer(("", port), CustomHandler) as httpd:
        url = f"http://localhost:{port}/network_builder.html"
        print("🧠 LeibNetz Network Builder")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"Server running at: {url}")
        print("Press Ctrl+C to stop the server")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # Open browser automatically
        if open_browser:
            print("\nOpening browser...")
            webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nShutting down server...")


def main():
    """Main entry point for the UI server CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Start the LeibNetz Network Builder UI server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to serve on (default: 8080)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )

    args = parser.parse_args()

    serve_ui(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
