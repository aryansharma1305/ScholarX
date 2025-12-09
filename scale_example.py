"""Example script for large-scale paper ingestion."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from manage_papers import batch_add_papers

# Example: Add papers on multiple research topics
if __name__ == "__main__":
    print("=" * 70)
    print("ScholarX - Large Scale Paper Ingestion")
    print("=" * 70)
    
    # Define topics for your research domain
    topics = [
        "transformer architecture",
        "attention mechanism",
        "neural networks",
        "deep learning",
        "machine learning",
        "computer vision",
        "natural language processing",
        "reinforcement learning",
    ]
    
    papers_per_topic = 15  # Adjust based on your needs
    
    print(f"\nWill add {len(topics)} topics × {papers_per_topic} papers = ~{len(topics) * papers_per_topic} papers")
    print("This may take 30-60 minutes depending on paper sizes")
    
    confirm = input("\nProceed? (yes/no): ").strip().lower()
    
    if confirm == "yes":
        successful, failed = batch_add_papers(topics, papers_per_topic)
        print(f"\n✅ Complete! Added {successful} papers, {failed} failed")
    else:
        print("Cancelled")

