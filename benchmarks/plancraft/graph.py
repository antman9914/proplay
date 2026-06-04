"""
Workflow Graph — planning-level memory structure for pre-play.

Nodes: workflow sub-routines (name + site + content)
Edges: observed transitions between workflows within a completed task
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


INITIAL_RELIABILITY: float = 0.5


@dataclass
class WorkflowNode:
    node_id: str
    site: str
    name: str     # e.g. "Find all orders"
    content: str  # full workflow steps text


@dataclass
class WorkflowEdge:
    edge_id: str
    source_id: str
    target_id: str
    count: int = 0
    success_count: float = 0.0
    emb_weighted_sum: Optional[List[float]] = None  # sum of reward * task_embedding
    emb_weight_sum: float = 0.0                     # sum of rewards (normalizer)

    @property
    def reliability(self) -> float:
        if self.count == 0:
            return INITIAL_RELIABILITY
        return max(0.0, min(1.0, self.success_count / self.count))


class WorkflowGraph:
    def __init__(self) -> None:
        self.nodes: Dict[str, WorkflowNode] = {}
        self.edges: Dict[str, WorkflowEdge] = {}
        self._site_nodes: Dict[str, List[str]] = {}
        self._edge_index: Dict[Tuple[str, str], str] = {}

    # ── Node management ──────────────────────────────────────────────────────

    def add_node(self, site: str, name: str, content: str) -> str:
        node_id = f"{site}_{uuid.uuid4().hex[:6]}"
        self.nodes[node_id] = WorkflowNode(
            node_id=node_id, site=site, name=name, content=content
        )
        self._site_nodes.setdefault(site, []).append(node_id)
        return node_id

    def get_nodes_for_site(self, site: str) -> List[WorkflowNode]:
        return [self.nodes[nid] for nid in self._site_nodes.get(site, [])]

    def node_by_name(self, site: str, name: str) -> Optional[WorkflowNode]:
        for node in self.get_nodes_for_site(site):
            if node.name.lower().strip() == name.lower().strip():
                return node
        return None

    # ── Edge management ──────────────────────────────────────────────────────

    def get_or_create_edge(self, src_id: str, tgt_id: str) -> str:
        key = (src_id, tgt_id)
        if key in self._edge_index:
            return self._edge_index[key]
        edge_id = uuid.uuid4().hex[:8]
        self.edges[edge_id] = WorkflowEdge(
            edge_id=edge_id, source_id=src_id, target_id=tgt_id
        )
        self._edge_index[key] = edge_id
        return edge_id

    def record_transition(
        self, src_id: str, tgt_id: str, score: float, task_embedding=None
    ) -> None:
        """score is the normalized episode reward (0–1); negatives are clipped to 0.

        task_embedding: plain Python list of floats (unit-normalized). When provided
        and score > 0, the embedding is accumulated as a reward-weighted sum so that
        get_task_reliability() can later compute task-specific reliability scores.
        """
        edge_id = self.get_or_create_edge(src_id, tgt_id)
        edge = self.edges[edge_id]
        edge.count += 1
        w = max(0.0, score)
        edge.success_count += w
        if task_embedding is not None and w > 0:
            if edge.emb_weighted_sum is None:
                edge.emb_weighted_sum = [v * w for v in task_embedding]
            else:
                s = edge.emb_weighted_sum
                for k in range(len(s)):
                    s[k] += task_embedding[k] * w
            edge.emb_weight_sum += w

    def get_task_reliability(self, edge: "WorkflowEdge", task_embedding) -> float:
        """Cosine similarity between task_embedding and the reward-weighted centroid.

        Falls back to global edge.reliability when no embedding data exists.
        task_embedding must be a unit-normalized list of floats.
        """
        if edge.emb_weighted_sum is None or edge.emb_weight_sum == 0:
            return edge.reliability
        dot = sum(a * b for a, b in zip(task_embedding, edge.emb_weighted_sum))
        mag = sum(v * v for v in edge.emb_weighted_sum) ** 0.5
        if mag == 0:
            return edge.reliability
        return max(0.0, min(1.0, dot / mag))

    # ── Serialization ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> "WorkflowGraph":
        graph = cls()
        data = json.loads(Path(path).read_text())
        for n in data["nodes"]:
            node = WorkflowNode(**n)
            graph.nodes[node.node_id] = node
            graph._site_nodes.setdefault(node.site, []).append(node.node_id)
        for e in data["edges"]:
            e.pop("reliability", None)
            # emb_weighted_sum / emb_weight_sum may be absent in older graph files
            edge = WorkflowEdge(**e)
            graph.edges[edge.edge_id] = edge
            graph._edge_index[(edge.source_id, edge.target_id)] = edge.edge_id
        return graph

    @classmethod
    def load_or_create(cls, path: str, workflow_path: str, site: str) -> "WorkflowGraph":
        """Load existing graph or bootstrap from workflow file."""
        if Path(path).exists():
            graph = cls.load(path)
            # add any new workflows not yet in graph
            existing_names = {n.name for n in graph.get_nodes_for_site(site)}
            for name, content in parse_workflows(workflow_path):
                if name not in existing_names:
                    graph.add_node(site=site, name=name, content=content)
        else:
            graph = cls()
            for name, content in parse_workflows(workflow_path):
                graph.add_node(site=site, name=name, content=content)
        return graph

    def to_dict(self) -> dict:
        edge_dicts = []
        for e in self.edges.values():
            d = {
                "edge_id": e.edge_id,
                "source_id": e.source_id,
                "target_id": e.target_id,
                "count": e.count,
                "success_count": e.success_count,
                "reliability": round(e.reliability, 3),
            }
            if e.emb_weighted_sum is not None:
                d["emb_weighted_sum"] = [round(v, 6) for v in e.emb_weighted_sum]
                d["emb_weight_sum"] = e.emb_weight_sum
            edge_dicts.append(d)
        return {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "site": n.site,
                    "name": n.name,
                    "content": n.content,
                }
                for n in self.nodes.values()
            ],
            "edges": edge_dicts,
        }

    def summary(self) -> str:
        sites = {n.site for n in self.nodes.values()}
        return (
            f"WorkflowGraph: {len(self.nodes)} nodes, {len(self.edges)} edges, "
            f"sites: {sorted(sites)}"
        )


# ── Workflow file parsing ─────────────────────────────────────────────────────

def parse_workflows(workflow_path: str) -> List[Tuple[str, str]]:
    """
    Parse 'Workflow N: <name>' blocks from a workflow file.
    Looks for a '## Summary Workflows' section first; falls back to
    scanning the whole file.
    Returns list of (name, content) tuples.
    """
    path = Path(workflow_path)
    if not path.exists():
        return []
    content = path.read_text()

    # prefer the summarised section if present
    summary_match = re.search(r'##\s*Summary Workflows\s*\n', content, re.I)
    if summary_match:
        content = content[summary_match.end():]

    pattern = re.compile(r'^Workflow\s+\d+:\s+(.+)$', re.MULTILINE)
    matches = list(pattern.finditer(content))
    if not matches:
        return []

    workflows = []
    for i, m in enumerate(matches):
        name = re.sub(r'\*+', '', m.group(1)).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        wf_content = content[start:end].strip()
        workflows.append((name, wf_content))

    return workflows
