"""Tests for the LeibNetz UI server module."""

import http.client
from pathlib import Path
import subprocess
import time

import pytest


def test_ui_files_exist():
    """Test that UI files exist in the expected location."""
    ui_dir = Path(__file__).parent.parent / "src" / "leibnetz" / "ui"
    assert ui_dir.exists(), "UI directory should exist"
    assert (ui_dir / "__init__.py").exists(), "UI __init__.py should exist"
    assert (ui_dir / "server.py").exists(), "UI server.py should exist"
    assert (
        ui_dir / "network_builder.html"
    ).exists(), "network_builder.html should exist"
    assert (ui_dir / "network_builder.js").exists(), "network_builder.js should exist"


def test_ui_server_starts():
    """Test that the UI server can start and serve the HTML file."""
    import random

    # Use random port to avoid conflicts in parallel tests
    port = random.randint(9000, 9999)

    # Start the server in a subprocess
    proc = subprocess.Popen(
        ["python", "-m", "leibnetz.ui.server", "--port", str(port), "--no-browser"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=Path(__file__).parent.parent,
    )

    # Give the server time to start and retry connection
    max_retries = 15
    retry_delay = 0.5
    connected = False

    for i in range(max_retries):
        time.sleep(retry_delay)
        if proc.poll() is not None:
            # Process died, capture error
            stdout, stderr = proc.communicate()
            pytest.fail(
                f"Server process died. stdout: {stdout.decode()}, stderr: {stderr.decode()}"
            )
        try:
            conn = http.client.HTTPConnection("localhost", port, timeout=2)
            conn.request("GET", "/network_builder.html")
            response = conn.getresponse()
            if response.status == 200:
                connected = True
                conn.close()
                break
            conn.close()
        except (ConnectionRefusedError, OSError):
            if i == max_retries - 1:
                # Last attempt failed
                pass
            continue

    try:
        # Check if the server is running
        assert proc.poll() is None, "Server process should be running"
        assert connected, "Server should accept connections within timeout"

        # Try to fetch the HTML file
        conn = http.client.HTTPConnection("localhost", port, timeout=5)
        try:
            conn.request("GET", "/network_builder.html")
            response = conn.getresponse()
            assert response.status == 200, "Server should return 200 OK"

            content = response.read().decode("utf-8")
            assert "LeibNetz Network Builder" in content, "HTML should contain title"
            assert "network_builder.js" in content, "HTML should reference JS file"
        finally:
            conn.close()

        # Try to fetch the JavaScript file
        conn = http.client.HTTPConnection("localhost", port, timeout=5)
        try:
            conn.request("GET", "/network_builder.js")
            response = conn.getresponse()
            assert response.status == 200, "Server should return 200 OK for JS"

            content = response.read().decode("utf-8")
            assert "NetworkBuilder" in content, "JS should contain NetworkBuilder class"
        finally:
            conn.close()

    finally:
        # Clean up: terminate the server
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def test_ui_module_imports():
    """Test that the UI module can be imported."""
    from leibnetz.ui import serve_ui

    assert callable(serve_ui), "serve_ui should be a callable function"


def test_html_contains_all_node_types():
    """Test that the HTML file includes all available node types."""
    ui_dir = Path(__file__).parent.parent / "src" / "leibnetz" / "ui"
    html_content = (ui_dir / "network_builder.html").read_text()

    node_types = [
        "ConvPassNode",
        "ResampleNode",
        "ConvResampleNode",
        "AdditiveAttentionGateNode",
        "WrapperNode",
    ]

    for node_type in node_types:
        assert node_type in html_content, f"{node_type} should be in the HTML palette"


def test_js_contains_node_definitions():
    """Test that the JavaScript file defines all node types."""
    ui_dir = Path(__file__).parent.parent / "src" / "leibnetz" / "ui"
    js_content = (ui_dir / "network_builder.js").read_text()

    node_types = [
        "ConvPassNode",
        "ResampleNode",
        "ConvResampleNode",
        "AdditiveAttentionGateNode",
        "WrapperNode",
    ]

    for node_type in node_types:
        assert node_type in js_content, f"{node_type} should be defined in JavaScript"

    # Check for key methods
    assert "addNode" in js_content, "addNode method should be defined"
    assert "addConnection" in js_content, "addConnection method should be defined"
    assert "generateCode" in js_content, "generateCode method should be defined"
    assert "exportNetwork" in js_content, "exportNetwork method should be defined"
    assert "importNetwork" in js_content, "importNetwork method should be defined"
