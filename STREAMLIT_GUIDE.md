# Streamlit App Guide

## 🚀 Quick Start

### 1. Install Streamlit (if not already installed)
```bash
pip install streamlit
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📱 Features

### Tab 1: Ask Question
- Enter any research question
- System automatically fetches papers from APIs
- Displays answer with citations
- Shows context chunks used

### Tab 2: Search Papers
- **By Topic**: Search for papers on any topic
- **By Author**: Find papers by specific authors
- **By Year**: Filter papers by publication year

### Tab 3: Collection
- View collection statistics
- List all papers in collection
- Refresh statistics

### Tab 4: About
- Information about ScholarX
- Features and APIs used

## ⚙️ Sidebar Settings

- **Papers to fetch per query**: 1-10 (default: 5)
- **Top K chunks**: 3-15 (default: 5)
- **Enhanced features**: Toggle hybrid search, re-ranking

## 💡 Usage Tips

1. **First Query**: May take 30-60 seconds (fetching + ingestion)
2. **Subsequent Queries**: Faster (papers already cached)
3. **New Topics**: Always fetches fresh papers
4. **Author Search**: Searches within collection (not APIs)

## 🎯 Example Queries

- "What is transformer architecture?"
- "How does attention mechanism work?"
- "Explain neural machine translation"
- "What are the latest advances in deep learning?"

## 📊 What You'll See

- **Answer**: Generated response with citations
- **Citations**: Papers used with relevance scores
- **Context Chunks**: Source text used for answer
- **Statistics**: Collection size and distribution

Enjoy testing ScholarX! 🎉

