"""Compatibility shim — re-exports from the local graph.py (plancraft copy)."""
from benchmarks.plancraft.graph import (  # noqa: F401
    WorkflowGraph, WorkflowNode, WorkflowEdge,
    parse_workflows, INITIAL_RELIABILITY,
)
