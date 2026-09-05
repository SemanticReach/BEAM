# BEAM Benchmark - Quick Start Guide

## 1. Install Dependencies

```bash
pip install pandas requests python-dotenv openai sentence-transformers langchain-openai nltk tqdm
```

## 2. Set Up Environment Variables

Create `.env` file:

```bash
HB_SERVER_URL=http://18.220.128.24:8000
HB_API_KEY=your_api_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
```

## 3. Run Ingestion

```bash
cd C:\Users\karin\jared\beam\BEAM

# Full ingest with wipe (recommended for fresh start)
python beam_ingest.py --chat-dir test_chats/100K/1 --chat-id 100K_1 --wipe

# Ingest without wiping (append mode)
python beam_ingest.py --chat-dir test_chats/100K/1 --chat-id 100K_1

# Evaluate only (skip ingestion)
python beam_ingest.py --chat-dir test_chats/100K/1 --chat-id 100K_1 --eval-only
```

## 4. Run Evaluation with DeepSeek Judge

```bash
python eval_beam_deepseek.py
```

## 5. Check Results

Results are saved to:

- `results/1/beam_scores_deepseek.json` - Detailed scores
- `results/beam_summary.txt` - Human-readable summary

## 6. (Optional) Run Official BEAM Evaluator

```bash
python -m src.evaluation.run_evaluation --input_directory results/ --chat_size 100K --start_index 0 --end_index 1 --max_workers 1 --allowed_result_files beam_results_100K_1_*.json
```

## Expected Output

```text
✅ Chat ingested: 188 rows → beam_chat_100K_1
✅ QA ingested: 20 rows → beam_qa_100K_1
✅ Results saved to: results\1\beam_results_*.json

============================================================
  📊 BEAM RESULTS (DeepSeek Judge)
============================================================
  abstention                    2/2  100.0%  (avg: 10.0/10)
  contradiction_resolution      2/2  100.0%  (avg: 10.0/10)
  instruction_following         2/2  100.0%  (avg: 10.0/10)
  preference_following          2/2  100.0%  (avg: 10.0/10)
  ...
============================================================
  OVERALL                      20/20  100.0%  (avg: 9.8/10)
============================================================
```

## Troubleshooting

- **Missing API key**: Ensure `.env` file has `DEEPSEEK_API_KEY` set
- **File not found**: Make sure you're in the `BEAM` directory
- **Module errors**: Run `pip install -r requirements.txt`
- **Results not saving**: Check `results/1/` directory exists