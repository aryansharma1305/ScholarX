"""View saved query results."""
import sys
import json
import glob
from pathlib import Path
from datetime import datetime

def list_results():
    """List all saved query results."""
    results = glob.glob("query_result_*.json")
    if not results:
        print("No saved results found.")
        return []
    
    results.sort(reverse=True)  # Newest first
    return results


def view_result(filename: str):
    """View a specific result file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        print("=" * 70)
        print(f"Query Result: {filename}")
        print("=" * 70)
        print(f"Query: {result['query']}")
        print(f"Timestamp: {result.get('timestamp', 'Unknown')}")
        
        print("\n" + "=" * 70)
        print("ANSWER")
        print("=" * 70)
        print(result['answer'])
        
        print("\n" + "=" * 70)
        print("CITATIONS")
        print("=" * 70)
        for i, citation in enumerate(result.get('citations', []), 1):
            print(f"{i}. Paper: {citation.get('paper_id', 'Unknown')}, "
                  f"Chunk: {citation.get('chunk_index', 'Unknown')}, "
                  f"Score: {citation.get('score', 0):.4f}")
        
        print("\n" + "=" * 70)
        print("CONTEXT CHUNKS")
        print("=" * 70)
        for i, chunk in enumerate(result.get('context_chunks', []), 1):
            print(f"\n[Chunk {i}]")
            print(f"Paper: {chunk.get('paper_id', 'Unknown')}")
            print(f"Chunk Index: {chunk.get('chunk_index', 'Unknown')}")
            print(f"Score: {chunk.get('score', 0):.4f}")
            print(f"Text: {chunk.get('text', '')[:300]}...")
        
    except Exception as e:
        print(f"Error reading file: {e}")


def main():
    """Main interface."""
    print("=" * 70)
    print("ScholarX - View Query Results")
    print("=" * 70)
    
    results = list_results()
    
    if not results:
        return
    
    print(f"\nFound {len(results)} saved results:\n")
    for i, filename in enumerate(results, 1):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            query = data.get('query', 'Unknown query')
            timestamp = data.get('timestamp', 'Unknown time')
            print(f"{i}. {filename}")
            print(f"   Query: {query[:50]}...")
            print(f"   Time: {timestamp}")
        except:
            print(f"{i}. {filename}")
    
    print("\nOptions:")
    print("1. View a specific result (enter number)")
    print("2. View all results")
    print("3. Exit")
    
    choice = input("\nEnter choice: ").strip()
    
    if choice == "1":
        num = input("Enter result number: ").strip()
        try:
            idx = int(num) - 1
            if 0 <= idx < len(results):
                view_result(results[idx])
            else:
                print("Invalid number")
        except:
            print("Invalid input")
    
    elif choice == "2":
        for filename in results:
            view_result(filename)
            print("\n" + "=" * 70 + "\n")
    
    elif choice == "3":
        print("Goodbye!")


if __name__ == "__main__":
    main()

