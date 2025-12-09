"""Interactive query interface for the RAG pipeline."""
import sys
import json
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent))

from main import query_rag
from config.settings import settings
from config.chroma_client import get_collection
from utils.logger import get_logger

logger = get_logger(__name__)

# Set to free mode
settings.embedding_provider = "sentence-transformers"
settings.llm_provider = "simple"


def display_result(result: dict, save_to_file: bool = True):
    """Display query result in a nice format."""
    print("\n" + "=" * 70)
    print("📝 ANSWER")
    print("=" * 70)
    print(result["answer"])
    
    print("\n" + "=" * 70)
    print("📚 CITATIONS")
    print("=" * 70)
    for i, citation in enumerate(result["citations"], 1):
        print(f"{i}. Paper ID: {citation['paper_id']}")
        print(f"   Chunk Index: {citation['chunk_index']}")
        print(f"   Relevance Score: {citation.get('score', 0):.4f}")
    
    print("\n" + "=" * 70)
    print("📄 CONTEXT CHUNKS")
    print("=" * 70)
    for i, chunk in enumerate(result["context_chunks"], 1):
        print(f"\n[Chunk {i}]")
        print(f"Paper ID: {chunk['paper_id']}")
        print(f"Chunk Index: {chunk['chunk_index']}")
        print(f"Score: {chunk.get('score', 0):.4f}")
        print(f"Text Preview: {chunk['text'][:200]}...")
    
    # Save to file
    if save_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"query_result_{timestamp}.json"
        
        output = {
            "query": result["query"],
            "timestamp": timestamp,
            "answer": result["answer"],
            "citations": result["citations"],
            "context_chunks": result["context_chunks"]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Result saved to: {filename}")


def show_stats():
    """Show collection statistics."""
    collection = get_collection()
    count = collection.count()
    print(f"\n📊 Collection Statistics:")
    print(f"   Total chunks stored: {count}")
    
    if count > 0:
        # Get sample to see paper IDs
        sample = collection.get(limit=min(100, count))
        if sample.get("metadatas"):
            paper_ids = set()
            for meta in sample["metadatas"]:
                if "paper_id" in meta:
                    paper_ids.add(meta["paper_id"])
            print(f"   Unique papers: {len(paper_ids)}")
            print(f"   Sample paper IDs: {', '.join(list(paper_ids)[:5])}")


def main():
    """Interactive query interface with advanced modes."""
    print("=" * 70)
    print("ScholarX RAG Pipeline - Interactive Query")
    print("=" * 70)
    print("\n💡 You can ask questions about research papers!")
    print("   The system will search through ingested papers and provide answers.")
    print("\n   Commands:")
    print("   - 'exit' or 'quit' to quit")
    print("   - 'stats' to see collection info")
    print("   - 'modes' to see available RAG modes")
    print("=" * 70)
    
    show_stats()
    
    while True:
        print("\n" + "-" * 70)
        query = input("\n❓ Your question (or command): ").strip()
        
        if not query:
            continue
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if query.lower() == 'stats':
            show_stats()
            continue
        
        if query.lower() == 'modes':
            print("\n📋 Available RAG Modes:")
            print("   1. default - Standard RAG")
            print("   2. concise - Short, direct answers")
            print("   3. detailed - Comprehensive answers")
            print("   4. explain - Simple explanations")
            print("   5. compare - Compare papers")
            print("   6. survey - Literature survey")
            print("\n   Usage: Type mode number before your query")
            print("   Example: '3 What is attention mechanism?'")
            continue
        
        # Check for mode prefix
        mode = "default"
        if query[0].isdigit():
            mode_num = int(query[0])
            query = query[1:].strip()
            mode_map = {
                1: "default",
                2: "concise",
                3: "detailed",
                4: "explain",
                5: "compare",
                6: "survey"
            }
            mode = mode_map.get(mode_num, "default")
            if mode != "default":
                print(f"   Using mode: {mode}")
        
        print(f"\n🔍 Searching for: '{query}'...")
        print("   (This may take a few seconds)")
        
        try:
            from api.rag_modes import (
                rag_query_concise, rag_query_detailed, rag_query_explain_simple,
                rag_query_compare, rag_query_literature_survey
            )
            
            if mode == "concise":
                result = rag_query_concise(query)
            elif mode == "detailed":
                result = rag_query_detailed(query)
            elif mode == "explain":
                result = rag_query_explain_simple(query)
            elif mode == "compare":
                result = rag_query_compare(query)
            elif mode == "survey":
                result = rag_query_literature_survey(query)
            else:
                result = query_rag(
                    query=query,
                    top_k=5,
                    fetch_papers=True,
                    use_enhanced=True
                )
            
            display_result(result, save_to_file=True)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

