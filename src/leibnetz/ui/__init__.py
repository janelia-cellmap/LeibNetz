"""
Browser-based UI for constructing LeibNetz networks visually.

This module provides a web-based interface for visually designing and
constructing neural network architectures using LeibNetz nodes.
"""

from .server import main, serve_ui

__all__ = ["serve_ui", "main"]
