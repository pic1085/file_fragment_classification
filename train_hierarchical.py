"""
Compatibility entrypoint.
Prefer: python scripts/train_hierarchical.py
"""
import runpy

runpy.run_module("scripts.train_hierarchical", run_name="__main__")
