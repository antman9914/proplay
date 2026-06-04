"""Compatibility shim — re-exports from the local graph.py (taubench copy)."""
from benchmarks.taubench.graph import (  # noqa: F401
    WorkflowGraph, WorkflowNode, WorkflowEdge,
    parse_workflows, INITIAL_RELIABILITY,
)
