"""
Citation-graph-aware retrieval boosting for ScholarX.

Signal pipeline (applied additively, all stored as chunk metadata):
  1. Neighbourhood overlap fraction  — hits/num_anchors (fixed denominator)
  2. Hop discount                    — 1.0 for 1-hop, 0.5 for 2-hop exclusive
  3. MIN_CITATION_ANCHOR_HITS filter — skip boost below threshold (ablation axis)
  4. Temporal decay                  — recency weight, stored for auditability

Ablation axes exposed via settings / environment:
  CITATION_BOOST_WEIGHT      — magnitude (0 = off)
  MIN_CITATION_ANCHOR_HITS   — minimum anchor overlap required (1 = any, 2 = multi-confirm)
  USE_2HOP_CITATION          — extend neighbourhood one extra hop at 0.5× weight
"""
from __future__ import annotations

import datetime
import time
from functools import lru_cache
from typing import Dict, List, Set, Tuple

from config.settings import settings
from utils.logger import get_logger
from vectorstore.query import QueryResult

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ANCHOR_PAPERS = 5          # top-N unique papers used as citation anchors
_2HOP_CAP = 8               # max 1-hop neighbours expanded for 2-hop (avoids explosion)
_2HOP_DECAY = 0.5           # 2-hop boost is this fraction of 1-hop boost
_API_DELAY = 0.15 if settings.semantic_scholar_api_key else 1.05

_CURRENT_YEAR = datetime.datetime.now().year


# ---------------------------------------------------------------------------
# Neighbourhood fetching (cached per paper_id for the process lifetime)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=512)
def _fetch_1hop_cached(paper_id: str) -> Tuple[str, ...]:
    """
    Return the 1-hop citation neighbourhood for a paper.
    Cached — each paper_id is fetched at most once per process.
    Returns a tuple (hashable, lru_cache compatible).
    """
    return tuple(_fetch_1hop_uncached(paper_id))


def _fetch_1hop_uncached(paper_id: str) -> Set[str]:
    """
    Hit Semantic Scholar for papers that cite OR are cited by `paper_id`.
    Returns a set of neighbour paper_ids (excluding the anchor itself).
    """
    try:
        from ingestion.semantic_scholar_enhanced import (
            get_paper_citations,
            get_paper_references,
        )

        neighbours: Set[str] = set()

        # Incoming edges (papers that cite this paper)
        for citing in get_paper_citations(paper_id, limit=50).get("data", []):
            pid = citing.get("paper_id")
            if pid and pid != paper_id:
                neighbours.add(pid)

        time.sleep(_API_DELAY)

        # Outgoing edges (papers this paper cites)
        for ref in get_paper_references(paper_id, limit=50).get("data", []):
            pid = ref.get("paper_id")
            if pid and pid != paper_id:
                neighbours.add(pid)

        logger.debug("1-hop neighbourhood for %s: %d neighbours", paper_id, len(neighbours))
        return neighbours

    except Exception as exc:
        logger.warning("Could not fetch neighbourhood for %s: %s", paper_id, exc)
        return set()


def _get_neighbourhood(paper_id: str, use_2hop: bool = False) -> Tuple[Set[str], Set[str]]:
    """
    Return (hop1_set, hop2_exclusive_set) for a paper.

    hop2_exclusive_set contains only papers reachable at exactly 2 hops
    (i.e., NOT already in hop1_set).  If use_2hop is False, hop2 is empty.

    The cap of _2HOP_CAP on hop1 expansion prevents the neighbourhood
    from exploding — a highly-cited paper might have thousands of 1-hop
    neighbours, each with thousands of their own.
    """
    hop1 = set(_fetch_1hop_cached(paper_id))
    if not use_2hop:
        return hop1, set()

    hop2: Set[str] = set()
    for pid in list(hop1)[:_2HOP_CAP]:
        hop2.update(_fetch_1hop_cached(pid))
        time.sleep(_API_DELAY * 0.5)   # shorter delay for 2-hop expansion

    hop2_exclusive = hop2 - hop1 - {paper_id}
    logger.debug(
        "2-hop neighbourhood for %s: %d exclusive 2-hop neighbours", paper_id, len(hop2_exclusive)
    )
    return hop1, hop2_exclusive


# ---------------------------------------------------------------------------
# Neighbourhood map construction
# ---------------------------------------------------------------------------

def build_citation_neighbourhood(
    anchor_paper_ids: List[str],
    use_2hop: bool = False,
) -> Dict[str, Tuple[float, int, str]]:
    """
    Build a mapping of {neighbour_paper_id -> (raw_boost_score, hit_count, hop_label)}
    for the given anchor papers.

    Boost formula:
        boost = CITATION_BOOST_WEIGHT
                × (anchor_hit_count / num_anchors)    ← overlap fraction
                × hop_weight                           ← 1.0 for 1-hop, 0.5 for 2-hop

    The denominator is always num_anchors (fixed), NOT max(hits).
    This ensures the boost is a true count multiplier comparable across queries.

    Args:
        anchor_paper_ids: Paper IDs of the top-ranked retrieved papers.
        use_2hop: Whether to expand one extra hop at half weight.

    Returns:
        Dict mapping paper_id -> (boost_score, anchor_hit_count, hop_label).
        hit_count and hop_label are stored in chunk metadata for ablation analysis.
    """
    boost_weight = settings.citation_boost_weight
    num_anchors = len(anchor_paper_ids)
    if num_anchors == 0:
        return {}

    # Count how many anchors include each neighbour (per hop)
    hop1_hits: Dict[str, int] = {}
    hop2_hits: Dict[str, int] = {}

    for anchor_id in anchor_paper_ids:
        hop1, hop2_exclusive = _get_neighbourhood(anchor_id, use_2hop=use_2hop)
        for pid in hop1:
            hop1_hits[pid] = hop1_hits.get(pid, 0) + 1
        for pid in hop2_exclusive:
            hop2_hits[pid] = hop2_hits.get(pid, 0) + 1

    if not hop1_hits and not hop2_hits:
        return {}

    result: Dict[str, Tuple[float, int, str]] = {}

    # 1-hop entries (full weight)
    for pid, hits in hop1_hits.items():
        overlap = hits / num_anchors           # true count multiplier
        boost = boost_weight * overlap * 1.0   # 1-hop weight = 1.0
        result[pid] = (boost, hits, "1-hop")

    # 2-hop exclusive entries (half weight); 1-hop takes priority if overlap
    for pid, hits in hop2_hits.items():
        if pid not in result:
            overlap = hits / num_anchors
            boost = boost_weight * overlap * _2HOP_DECAY
            result[pid] = (boost, hits, "2-hop")

    return result


# ---------------------------------------------------------------------------
# Temporal decay helper (stored as metadata, not baked in)
# ---------------------------------------------------------------------------

def _temporal_weight(paper_year: int | None) -> float:
    """
    Recency weight in [0.5, 1.0].
    Decays by 2% per year of age, floors at 0.5.

    Stored as chunk metadata so evaluation code can inspect it independently.
    """
    if paper_year is None:
        return 1.0
    age = max(0, _CURRENT_YEAR - int(paper_year))
    return max(0.5, 1.0 - age * 0.02)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def apply_citation_boost(
    results: List[QueryResult],
    top_anchor_k: int = _ANCHOR_PAPERS,
) -> List[QueryResult]:
    """
    Apply citation-graph-aware score boosting to a ranked list of chunks.

    Pipeline:
        1. Pick top `top_anchor_k` unique papers as anchors.
        2. Build 1-hop (and optionally 2-hop) neighbourhood via Semantic Scholar.
        3. For each retrieved chunk:
           a. Look up its paper in the neighbourhood map.
           b. Skip if anchor_hit_count < settings.min_citation_anchor_hits.
           c. Compute temporal_weight from paper year.
           d. boost = base_boost × temporal_weight.
           e. Store ALL intermediate values as chunk metadata for the ablation study.
        4. Re-sort and return.

    Args:
        results:      Ranked QueryResult list from hybrid/semantic search + reranker.
        top_anchor_k: How many top papers to use as citation anchors.

    Returns:
        Re-sorted list with citation boost metadata embedded in each chunk.
    """
    if not results:
        return results

    boost_weight = settings.citation_boost_weight
    if boost_weight <= 0:
        logger.debug("Citation boost disabled (CITATION_BOOST_WEIGHT=0), skipping.")
        return results

    min_hits = settings.min_citation_anchor_hits
    use_2hop = settings.use_2hop_citation

    # ── Step 1: Identify unique anchor papers ────────────────────────────────
    seen: Set[str] = set()
    anchor_ids: List[str] = []
    for chunk in results:
        if chunk.paper_id not in seen:
            seen.add(chunk.paper_id)
            anchor_ids.append(chunk.paper_id)
        if len(anchor_ids) >= top_anchor_k:
            break

    logger.info(
        "[CitBoost] %d anchor papers → building citation neighbourhood "
        "(use_2hop=%s, min_anchor_hits=%d)",
        len(anchor_ids), use_2hop, min_hits,
    )

    # ── Step 2: Build neighbourhood map ─────────────────────────────────────
    neighbourhood = build_citation_neighbourhood(anchor_ids, use_2hop=use_2hop)

    if not neighbourhood:
        logger.info("[CitBoost] Neighbourhood empty — no boost applied.")
        return results

    num_anchors = len(anchor_ids)
    max_hits = max(v[1] for v in neighbourhood.values())
    logger.info(
        "[CitBoost] Neighbourhood: %d papers eligible "
        "(anchors=%d, max_overlap=%d/%d=%.2f)",
        len(neighbourhood), num_anchors, max_hits, num_anchors,
        max_hits / num_anchors,
    )

    # ── Step 3: Apply overlap-weighted, temporally-decayed boost ────────────
    boosted: List[QueryResult] = []
    for chunk in results:
        base_meta = chunk.metadata if chunk.metadata is not None else {}
        entry = neighbourhood.get(chunk.paper_id)

        if entry is not None:
            base_boost, anchor_hit_count, hop_label = entry
        else:
            base_boost, anchor_hit_count, hop_label = 0.0, 0, "none"

        overlap_fraction = anchor_hit_count / num_anchors

        # ── MIN_CITATION_ANCHOR_HITS filter (ablation axis #3) ──────────────
        passes_threshold = anchor_hit_count >= min_hits
        if not passes_threshold:
            base_boost = 0.0  # suppress boost, but still record metadata

        # ── Temporal decay (stored as metadata, applied to boost) ────────────
        paper_year = base_meta.get("year") or base_meta.get("publication_year")
        temporal_weight = _temporal_weight(paper_year)
        citation_boost_applied = base_boost * temporal_weight

        new_score = chunk.score + citation_boost_applied

        boosted.append(
            QueryResult(
                chunk_id=chunk.chunk_id,
                paper_id=chunk.paper_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                score=new_score,
                metadata={
                    **base_meta,
                    # ── scores ──────────────────────────────────────────────
                    "base_score": chunk.score,
                    "citation_boosted_score": new_score,
                    # ── overlap signal ──────────────────────────────────────
                    "citation_anchor_hits": anchor_hit_count,
                    "citation_overlap_fraction": overlap_fraction,
                    "citation_hop": hop_label,
                    # ── threshold gate ──────────────────────────────────────
                    "citation_passes_threshold": passes_threshold,
                    # ── temporal decay ──────────────────────────────────────
                    "temporal_weight": temporal_weight,
                    # ── final applied boost (auditable) ─────────────────────
                    "citation_boost": citation_boost_applied,
                },
            )
        )

    # ── Step 4: Re-sort by final score ───────────────────────────────────────
    boosted.sort(key=lambda x: x.score, reverse=True)

    # Stage-level diagnostic logging (same numbers that go in the paper)
    boosted_chunks = [c for c in boosted if c.metadata.get("citation_boost", 0) > 0]
    unboosted_chunks = [c for c in boosted if c.metadata.get("citation_boost", 0) == 0]

    avg_score = lambda lst: sum(c.score for c in lst) / len(lst) if lst else 0.0

    logger.info(
        "[CitBoost] %d/%d chunks boosted (threshold=%d anchors, 2-hop=%s)",
        len(boosted_chunks), len(boosted), min_hits, use_2hop,
    )
    if boosted_chunks:
        avg_overlap = sum(
            c.metadata["citation_overlap_fraction"] for c in boosted_chunks
        ) / len(boosted_chunks)
        avg_temporal = sum(
            c.metadata["temporal_weight"] for c in boosted_chunks
        ) / len(boosted_chunks)
        logger.info(
            "[CitBoost] avg overlap fraction (boosted): %.3f | avg temporal weight: %.3f",
            avg_overlap, avg_temporal,
        )
    logger.info(
        "[CitBoost] score delta — boosted avg: %.4f  vs  unboosted avg: %.4f  (Δ=%.4f)",
        avg_score(boosted_chunks),
        avg_score(unboosted_chunks),
        avg_score(boosted_chunks) - avg_score(unboosted_chunks),
    )

    return boosted
