# debug_10m_failures.py
import json
from pathlib import Path

results_dir = Path("C:/Users/karin/jared/beam/BEAM/results")

for category in ["instruction_following", "preference_following"]:
    print(f"\n{'='*80}")
    print(f"🔴 {category.upper()} FAILURES")
    print('='*80)
    
    for result_file in sorted(results_dir.glob("beam_results_10M_*.json")):
        with open(result_file) as f:
            data = json.load(f)
        
        if category in data.get("results", {}):
            questions = data["results"][category].get("questions", [])
            for q in questions:
                if not q.get("score", {}).get("pass", False):
                    print(f"\n📁 {result_file.name}")
                    print(f"Question: {q['question'][:200]}")
                    print(f"Expected: {q['ideal_answer'][:200]}")
                    print(f"Got:      {q['generated'][:200]}")
                    print(f"Score:    {q['score'].get('rubric_score')}")
                    print(f"Rubric:   {q['score'].get('rubric_values')}")
                    print(f"Hits:     {q['score'].get('rubric_hits')}")
                    print("-"*80)