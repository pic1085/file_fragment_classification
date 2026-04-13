"""
Compatibility entrypoint.
Prefer: python scripts/extract_features.py
"""
import runpy

runpy.run_module("scripts.extract_features", run_name="__main__")
