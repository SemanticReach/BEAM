# analyze_10m_results.py
import json
from pathlib import Path
from collections import defaultdict

def analyze_10m_results(results_dir: Path = Path(".")):
    """
    Analyze all 10M result JSON files and calculate overall accuracy.
    """
    # Find all 10M result files
    result_files = list(results_dir.glob("beam_results_10M_*.json"))
    
    if not result_files:
        print("No 10M result files found!")
        return
    
    print(f"📊 Found {len(result_files)} 10M result files\n")
    print("=" * 80)
    
    # Overall totals
    total_passed_all = 0
    total_questions_all = 0
    
    # Per-category totals
    category_totals = defaultdict(lambda: {"passed": 0, "total": 0})
    
    # Per-chat results
    chat_results = []
    
    for result_file in sorted(result_files):
        with open(result_file) as f:
            data = json.load(f)
        
        chat_id = data.get("chat_id", "unknown")
        
        # Get overall stats from the file
        overall = data.get("overall", {})
        passed = overall.get("passed", 0)
        total = overall.get("total", 0)
        
        total_passed_all += passed
        total_questions_all += total
        
        # Per-category breakdown
        results = data.get("results", {})
        for category, cat_data in results.items():
            cat_passed = cat_data.get("passed", 0)
            cat_total = cat_data.get("total", 0)
            category_totals[category]["passed"] += cat_passed
            category_totals[category]["total"] += cat_total
        
        # Store chat result
        accuracy = passed / total if total > 0 else 0
        chat_results.append({
            "chat_id": chat_id,
            "passed": passed,
            "total": total,
            "accuracy": accuracy
        })
        
        print(f"Chat {chat_id:>3}: {passed:>3}/{total:<3} ({accuracy:.1%})")
    
    # Calculate overall accuracy
    overall_accuracy = total_passed_all / total_questions_all if total_questions_all > 0 else 0
    
    print("\n" + "=" * 80)
    print(f"\n📈 OVERALL 10M RESULTS:")
    print(f"   Total Passed:  {total_passed_all}")
    print(f"   Total Questions: {total_questions_all}")
    print(f"   Overall Accuracy: {overall_accuracy:.2%}")
    print(f"   Number of Chats: {len(chat_results)}")
    
    # Per-category breakdown
    print("\n" + "=" * 80)
    print("\n📊 PER-CATEGORY BREAKDOWN:")
    print(f"{'Category':<35} {'Passed':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 65)
    
    for category in sorted(category_totals.keys()):
        stats = category_totals[category]
        passed = stats["passed"]
        total = stats["total"]
        accuracy = passed / total if total > 0 else 0
        print(f"{category:<35} {passed:>8} {total:>8} {accuracy:>9.2%}")
    
    # Best and worst chats
    print("\n" + "=" * 80)
    print("\n🏆 BEST PERFORMING CHATS (Top 5):")
    best_chats = sorted(chat_results, key=lambda x: x["accuracy"], reverse=True)[:5]
    for i, chat in enumerate(best_chats, 1):
        print(f"   {i}. Chat {chat['chat_id']}: {chat['accuracy']:.1%} ({chat['passed']}/{chat['total']})")
    
    print("\n📉 WORST PERFORMING CHATS (Bottom 5):")
    worst_chats = sorted(chat_results, key=lambda x: x["accuracy"])[:5]
    for i, chat in enumerate(worst_chats, 1):
        print(f"   {i}. Chat {chat['chat_id']}: {chat['accuracy']:.1%} ({chat['passed']}/{chat['total']})")
    
    # Summary by category performance
    print("\n" + "=" * 80)
    print("\n🎯 CATEGORY PERFORMANCE SUMMARY:")
    
    # Find strongest and weakest categories
    category_accuracies = []
    for category, stats in category_totals.items():
        acc = stats["passed"] / stats["total"] if stats["total"] > 0 else 0
        category_accuracies.append((category, acc))
    
    category_accuracies.sort(key=lambda x: x[1], reverse=True)
    
    print("\n   Strongest Categories (Top 3):")
    for i, (cat, acc) in enumerate(category_accuracies[:3], 1):
        print(f"      {i}. {cat}: {acc:.1%}")
    
    print("\n   Weakest Categories (Bottom 3):")
    for i, (cat, acc) in enumerate(category_accuracies[-3:], 1):
        print(f"      {i}. {cat}: {acc:.1%}")
    
    # Save summary to file
    summary_file = results_dir / "10M_summary.json"
    summary = {
        "total_chats": len(chat_results),
        "total_passed": total_passed_all,
        "total_questions": total_questions_all,
        "overall_accuracy": overall_accuracy,
        "category_breakdown": {
            cat: {
                "passed": stats["passed"],
                "total": stats["total"],
                "accuracy": stats["passed"] / stats["total"] if stats["total"] > 0 else 0
            }
            for cat, stats in category_totals.items()
        },
        "per_chat_summary": chat_results
    }
    
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Summary saved to: {summary_file}")
    
    return summary

if __name__ == "__main__":
    # Run from the results directory
    results_path = Path("C:/Users/karin/jared/beam/BEAM/results")
    analyze_10m_results(results_path)