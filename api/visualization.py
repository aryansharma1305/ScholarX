"""Citation network visualization using NetworkX and Plotly."""
from typing import List, Dict, Optional, Set
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from config.chroma_client import get_collection
from api.citations import get_citation_info, find_citing_papers
from api.paper_api import get_paper_by_id
from rag.search_enhanced import get_related_papers
from utils.logger import get_logger

logger = get_logger(__name__)


def build_citation_graph(
    paper_ids: List[str],
    max_depth: int = 2,
    max_nodes: int = 50
) -> nx.DiGraph:
    """
    Build a citation graph from paper IDs.
    
    Args:
        paper_ids: List of paper IDs to include
        max_depth: Maximum depth to traverse (1 = direct citations, 2 = citations of citations)
        max_nodes: Maximum number of nodes in graph
        
    Returns:
        NetworkX directed graph
    """
    G = nx.DiGraph()
    collection = get_collection()
    visited = set()
    to_process = [(pid, 0) for pid in paper_ids]  # (paper_id, depth)
    
    logger.info(f"Building citation graph for {len(paper_ids)} papers, max_depth={max_depth}")
    
    while to_process and len(G.nodes) < max_nodes:
        current_paper_id, depth = to_process.pop(0)
        
        if current_paper_id in visited or depth > max_depth:
            continue
            
        visited.add(current_paper_id)
        
        # Get paper metadata
        paper_chunks = collection.get(
            where={"paper_id": current_paper_id},
            limit=1
        )
        
        if not paper_chunks.get("ids"):
            continue
            
        metadata = paper_chunks["metadatas"][0] if paper_chunks.get("metadatas") else {}
        title = metadata.get("title", "Unknown")
        authors = metadata.get("authors", "Unknown")
        year = metadata.get("year")
        citation_count = metadata.get("citation_count", 0)
        
        # Add node
        G.add_node(
            current_paper_id,
            title=title[:60] + "..." if len(title) > 60 else title,
            full_title=title,
            authors=authors,
            year=year,
            citation_count=citation_count,
            depth=depth
        )
        
        # Find related/citing papers (simulated citations)
        if depth < max_depth:
            try:
                # Get papers that cite this one (similar papers)
                citing_papers = find_citing_papers(current_paper_id, limit=5)
                
                for citing_paper in citing_papers:
                    cited_paper_id = citing_paper["paper_id"]
                    similarity = citing_paper.get("similarity_score", 0.0)
                    
                    # Add edge (citation relationship)
                    if cited_paper_id not in G.nodes:
                        # Get cited paper metadata
                        cited_chunks = collection.get(
                            where={"paper_id": cited_paper_id},
                            limit=1
                        )
                        
                        if cited_chunks.get("ids"):
                            cited_meta = cited_chunks["metadatas"][0] if cited_chunks.get("metadatas") else {}
                            G.add_node(
                                cited_paper_id,
                                title=(cited_meta.get("title", "Unknown")[:60] + "..." 
                                       if len(cited_meta.get("title", "")) > 60 
                                       else cited_meta.get("title", "Unknown")),
                                full_title=cited_meta.get("title", "Unknown"),
                                authors=cited_meta.get("authors", "Unknown"),
                                year=cited_meta.get("year"),
                                citation_count=cited_meta.get("citation_count", 0),
                                depth=depth + 1
                            )
                            
                            # Add to processing queue
                            if len(G.nodes) < max_nodes:
                                to_process.append((cited_paper_id, depth + 1))
                    
                    # Add edge with weight
                    G.add_edge(
                        current_paper_id,
                        cited_paper_id,
                        weight=similarity,
                        relationship="cites"
                    )
                    
            except Exception as e:
                logger.warning(f"Error finding citations for {current_paper_id}: {e}")
    
    logger.info(f"Built citation graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
    return G


def visualize_citation_network(
    paper_ids: List[str],
    max_depth: int = 2,
    max_nodes: int = 50,
    layout: str = "spring"
) -> go.Figure:
    """
    Create an interactive citation network visualization.
    
    Args:
        paper_ids: List of paper IDs to visualize
        max_depth: Maximum depth for citation traversal
        max_nodes: Maximum nodes in graph
        layout: Graph layout ('spring', 'circular', 'kamada_kawai', 'forceatlas2')
        
    Returns:
        Plotly figure object
    """
    # Build graph
    G = build_citation_graph(paper_ids, max_depth=max_depth, max_nodes=max_nodes)
    
    if len(G.nodes) == 0:
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="No citation data available. Add more papers to see the network.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Calculate layout positions
    if layout == "spring":
        pos = nx.spring_layout(G, k=2, iterations=50)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        try:
            pos = nx.kamada_kawai_layout(G)
        except:
            pos = nx.spring_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    # Prepare edge traces
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Edge info
        weight = G.edges[edge].get("weight", 0.0)
        edge_info.append({
            "from": G.nodes[edge[0]]["title"],
            "to": G.nodes[edge[1]]["title"],
            "weight": weight
        })
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines',
        name='Citations'
    )
    
    # Prepare node traces
    node_x = []
    node_y = []
    node_text = []
    node_sizes = []
    node_colors = []
    node_info = []
    
    for node_id in G.nodes():
        x, y = pos[node_id]
        node_x.append(x)
        node_y.append(y)
        
        node_data = G.nodes[node_id]
        title = node_data.get("title", "Unknown")
        authors = node_data.get("authors", "Unknown")
        year = node_data.get("year", "N/A")
        citations = node_data.get("citation_count", 0)
        depth = node_data.get("depth", 0)
        
        # Node label
        node_text.append(f"{title}<br>Authors: {authors}<br>Year: {year}")
        
        # Node size (based on citation count and depth)
        base_size = 15
        citation_bonus = min(citations / 10, 20)  # Max 20 extra
        depth_penalty = depth * 2  # Smaller for deeper nodes
        node_size = base_size + citation_bonus - depth_penalty
        node_sizes.append(max(10, node_size))
        
        # Node color (based on depth)
        if depth == 0:
            color = "#1f77b4"  # Blue for root papers
        elif depth == 1:
            color = "#ff7f0e"  # Orange for first-level citations
        else:
            color = "#2ca02c"  # Green for deeper citations
        
        node_colors.append(color)
        
        # Node info for hover
        node_info.append({
            "paper_id": node_id,
            "title": node_data.get("full_title", title),
            "authors": authors,
            "year": year,
            "citations": citations,
            "depth": depth
        })
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[G.nodes[n]["title"][:30] for n in G.nodes()],
        textposition="middle center",
        textfont=dict(size=8),
        hovertext=node_text,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white'),
            showscale=True,
            colorscale='Viridis',
            colorbar=dict(
                title="Depth",
                thickness=15,
                len=0.5,
                x=1.05
            )
        ),
        name='Papers',
        customdata=node_info
    )
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=f'Citation Network ({len(G.nodes)} papers, {len(G.edges)} citations)',
                x=0.5,
                font=dict(size=20)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Blue: Root papers | Orange: First-level citations | Green: Deeper citations",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.05,
                    xanchor="center",
                    font=dict(size=12, color="#666")
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=700
        )
    )
    
    return fig


def get_influential_papers(graph: nx.DiGraph, top_k: int = 10) -> List[Dict]:
    """
    Identify influential papers in the citation network.
    
    Uses PageRank algorithm to find most influential papers.
    
    Args:
        graph: NetworkX citation graph
        top_k: Number of top papers to return
        
    Returns:
        List of influential papers with scores
    """
    if len(graph.nodes) == 0:
        return []
    
    # Calculate PageRank
    try:
        pagerank = nx.pagerank(graph, max_iter=100)
    except:
        # Fallback to degree centrality
        pagerank = dict(graph.in_degree())
        total = sum(pagerank.values()) if pagerank.values() else 1
        pagerank = {k: v/total for k, v in pagerank.items()}
    
    # Sort by PageRank score
    influential = sorted(
        pagerank.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    
    results = []
    for paper_id, score in influential:
        node_data = graph.nodes[paper_id]
        results.append({
            "paper_id": paper_id,
            "title": node_data.get("full_title", "Unknown"),
            "authors": node_data.get("authors", "Unknown"),
            "year": node_data.get("year"),
            "influence_score": round(score, 4),
            "citation_count": node_data.get("citation_count", 0)
        })
    
    return results


def get_research_communities(graph: nx.DiGraph) -> Dict:
    """
    Identify research communities using community detection.
    
    Args:
        graph: NetworkX citation graph
        
    Returns:
        Dictionary with community information
    """
    if len(graph.nodes) < 3:
        return {"communities": [], "num_communities": 0}
    
    try:
        # Convert to undirected for community detection
        G_undirected = graph.to_undirected()
        
        # Use greedy modularity communities
        try:
            from networkx.algorithms import community
            communities = list(community.greedy_modularity_communities(G_undirected))
        except ImportError:
            # Fallback to simple connected components
            communities = list(nx.connected_components(G_undirected))
        except Exception:
            # Fallback to simple connected components
            communities = list(nx.connected_components(G_undirected))
        
        community_data = []
        for i, community in enumerate(communities):
            community_papers = []
            for paper_id in community:
                node_data = graph.nodes[paper_id]
                community_papers.append({
                    "paper_id": paper_id,
                    "title": node_data.get("full_title", "Unknown"),
                    "authors": node_data.get("authors", "Unknown")
                })
            
            community_data.append({
                "community_id": i,
                "size": len(community_papers),
                "papers": community_papers
            })
        
        return {
            "communities": community_data,
            "num_communities": len(communities)
        }
        
    except Exception as e:
        logger.error(f"Error detecting communities: {e}")
        return {"communities": [], "num_communities": 0}


def get_citation_statistics(graph: nx.DiGraph) -> Dict:
    """
    Get statistics about the citation network.
    
    Args:
        graph: NetworkX citation graph
        
    Returns:
        Dictionary with network statistics
    """
    if len(graph.nodes) == 0:
        return {}
    
    # Basic statistics
    num_nodes = len(graph.nodes)
    num_edges = len(graph.edges)
    
    # Degree statistics
    in_degrees = dict(graph.in_degree())
    out_degrees = dict(graph.out_degree())
    
    # Most cited papers
    most_cited = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Papers with most citations (outgoing)
    most_citing = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Network density
    density = nx.density(graph)
    
    # Average degree
    avg_in_degree = sum(in_degrees.values()) / num_nodes if num_nodes > 0 else 0
    avg_out_degree = sum(out_degrees.values()) / num_nodes if num_nodes > 0 else 0
    
    return {
        "num_papers": num_nodes,
        "num_citations": num_edges,
        "density": round(density, 4),
        "avg_in_degree": round(avg_in_degree, 2),
        "avg_out_degree": round(avg_out_degree, 2),
        "most_cited": [
            {
                "paper_id": pid,
                "title": graph.nodes[pid].get("title", "Unknown"),
                "in_degree": degree
            }
            for pid, degree in most_cited
        ],
        "most_citing": [
            {
                "paper_id": pid,
                "title": graph.nodes[pid].get("title", "Unknown"),
                "out_degree": degree
            }
            for pid, degree in most_citing
        ]
    }

