# Web Scraping Tutor

A scalable, fault-tolerant system for scraping and transforming Apache Jira issues into high-quality JSONL datasets for Large Language Model (LLM) training.

---

## 🚀 Overview
This project extracts public issue data from **Apache Jira**, cleans it, and converts it into structured LLM tasks such as:

- Summarization  
- Issue classification  
- Q&A generation  
- Resolution prediction  

The system is designed to be **robust**, **recoverable**, and **easily extensible**.

---

## 🎯 Features

### 🕸️ Web Scraping
- Scrapes issues from projects like **KAFKA**, **SPARK**, **HADOOP**
- Handles pagination, batching, rate limits
- Retry mechanism with exponential backoff
- Circuit breaker for repeated failures
- Transparent checkpointing system for resuming after interruption

### 🧹 Data Cleaning
- Removes Jira markup: `{code}`, `{quote}`, `{noformat}`
- Removes Jira links and user mentions
- Normalizes whitespace
- Extracts code blocks
- Truncates long text safely

### 🤖 LLM Task Generation
- Summarization  
- Classification  
- Q&A pairs  
- Resolution prediction  
- Assigns quality scores  
- Deduplicates using MD5 hashing  
- Produces **JSONL** output  

---

## 🏗️ Architecture

```
pipeline.py   →  Main orchestrator  
scraper.py    →  Handles API calls, rate limiting, retries, checkpointing  
transformer.py → Cleans, validates, and generates LLM tasks  
output/       → JSONL training datasets  
checkpoints/  → Saved progress for resume mode  
```

---

## 📂 Project Structure

```
.
├── pipeline.py
├── scraper.py
├── transformer.py
├── test_pipeline.py
├── config.yaml
├── output/
├── checkpoints/
└── README.md
```

---

## ⚙️ Installation

### Install dependencies
```
pip install -r requirements.txt
```

---

## ▶️ Running the Pipeline

### Full run
```
python pipeline.py --config config.yaml
```

### Resume from checkpoint
```
python pipeline.py --resume
```

### Scrape only one project
```
python pipeline.py --project KAFKA
```

### Change output directory
```
python pipeline.py --output ./my_output
```

---

## 📄 Output Format

Each JSONL line represents **one LLM training example**:

```json
{
  "task": "summarization",
  "input": "Issue description...",
  "output": "Summary...",
  "project": "KAFKA",
  "quality_score": 0.8,
  "metadata": {"issue_key": "KAFKA-1234"}
}
```

---

## 🧪 Testing

Run the full test suite:

```
python test_pipeline.py
```

Covers:

- RateLimiter  
- CircuitBreaker  
- CheckpointManager  
- TextCleaner  
- DataValidator  
- JSONL writing  
- Deduplication  
- Unicode handling  

---

## 🔐 Fault Tolerance Documentation

Your pipeline includes:

- **Rate limiting** to comply with API limits  
- **Exponential backoff** for retries  
- **Circuit breaker** to avoid repeated failed calls  
- **Checkpoint files** to resume progress  
- **Strict validation** to discard bad issues  
- **Per-issue isolation** (errors never stop the pipeline)

---

## 🚀 Future Improvements

- Async scraping with aiohttp  
- Docker containerization  
- Store raw data in Parquet  
- Add monitoring dashboard  
- Cloud-based storage integration (AWS/GCP/Azure)  
