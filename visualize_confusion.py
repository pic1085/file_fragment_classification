"""
Compatibility entrypoint.
Prefer: python scripts/visualize_confusion.py
"""
import runpy

runpy.run_module("scripts.visualize_confusion", run_name="__main__")
