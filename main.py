"""
Compatibility entrypoint.
Prefer: python scripts/main.py
"""
import runpy

runpy.run_module("scripts.main", run_name="__main__")
