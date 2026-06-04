"""Compatibility shim — re-exports from the shared proplay.graph module."""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from proplay.graph import (  # noqa: F401
    WorkflowGraph, WorkflowNode, WorkflowEdge,
    parse_workflows, INITIAL_RELIABILITY,
)
