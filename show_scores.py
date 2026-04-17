# show_scores.py - Quick script to display BEAM 10M percentages

import json
from pathlib import Path

results_dir = Path("results")

# Find all 10M result files from April 17 (latest runs)
result_files = list(results_dir.glob("beam_results_10M_*_20260417_*.json"))

if not result_files:
    print("No 10M result files found!")
    exit()

print("\n" + "=" * 70)
print("BEAM 10M RESULTS - April 17, 2026")
print("=" * 70)

total_passed_all = 0
total_questions_all = 0

# Store per-chat results
chat_results = []

for file in sorted(result_files):
    with open(file) as f:
        data = json.load(f)
    
    chat_id = data.get("chat_id", "unknown")
    overall = data.get("overall", {})
    passed = overall.get("passed", 0)
    total = overall.get("total", 0)
    accuracy = (passed / total * 100) if total > 0 else 0
    
    # Get category breakdown
    results = data.get("results", {})
    
    total_passed_all += passed
    total_questions_all += total
    
    chat_results.append({
        "chat_id": chat_id,
        "passed": passed,
        "total": total,
        "accuracy": accuracy,
        "results": results
    })
    
    print(f"\n{chat_id}: {accuracy:.1f}% ({passed}/{total})")

# Overall summary
print("\n" + "=" * 70)
print(f"OVERALL: {total_passed_all}/{total_questions_all} = {(total_passed_all/total_questions_all*100):.1f}%")
print("=" * 70)

# Per-category breakdown across all chats
print("\n" + "-" * 70)
print("PER-CATEGORY BREAKDOWN (across all 10 chats):")
print("-" * 70)

category_totals = {}

for chat in chat_results:
    for cat, cat_data in chat["results"].items():
        if cat not in category_totals:
            category_totals[cat] = {"passed": 0, "total": 0}
        category_totals[cat]["passed"] += cat_data.get("passed", 0)
        category_totals[cat]["total"] += cat_data.get("total", 0)

for cat in sorted(category_totals.keys()):
    stats = category_totals[cat]
    acc = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
    print(f"  {cat:<30} {stats['passed']:>3}/{stats['total']:<3} = {acc:>5.1f}%")

print("=" * 70)