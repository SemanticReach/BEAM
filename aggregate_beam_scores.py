import json
from pathlib import Path
from collections import defaultdict
import glob

def aggregate_beam_results(base_directory="results", chat_size="100K"):
    """
    Aggregate BEAM evaluation results from all chat directories.
    """
    
    all_scores = defaultdict(list)
    chat_scores_list = []
    
    # Find all evaluation files
    eval_files = list(Path(base_directory).glob("*/evaluation-*.json"))
    
    if not eval_files:
        print(f"❌ No evaluation files found in {base_directory}/")
        print("Make sure evaluations have completed.")
        return
    
    print(f"📊 Found {len(eval_files)} evaluation files\n")
    
    # Sort by chat number
    eval_files = sorted(eval_files, key=lambda x: int(x.parent.name))
    
    for eval_file in eval_files:
        chat_id = eval_file.parent.name
        
        with open(eval_file, 'r') as f:
            data = json.load(f)
        
        chat_scores = {}
        
        for category, items in data.items():
            category_scores = []
            for item in items:
                # Handle different possible score field names
                score = item.get("llm_judge_score") or item.get("score") or item.get("llm_score")
                if score is not None:
                    category_scores.append(float(score))
            
            if category_scores:
                avg_score = sum(category_scores) / len(category_scores)
                chat_scores[category] = {
                    "average": avg_score,
                    "scores": category_scores,
                    "count": len(category_scores)
                }
                all_scores[category].extend(category_scores)
        
        # Calculate overall for this chat
        if chat_scores:
            overall = sum(s["average"] for s in chat_scores.values()) / len(chat_scores)
            chat_scores["overall"] = overall
        
        chat_scores_list.append({
            "chat_id": chat_id,
            "scores": chat_scores
        })
    
    # Print results
    print("="*70)
    print(f"  📊 BEAM BENCHMARK RESULTS - {chat_size} CHATS")
    print("="*70)
    print(f"  Chats processed: {len(chat_scores_list)}")
    print()
    
    # Category averages
    print("  📈 PER-CATEGORY RESULTS:")
    print("-"*70)
    
    category_order = [
        "abstention", "contradiction_resolution", "event_ordering", 
        "information_extraction", "instruction_following", "knowledge_update",
        "multi_session_reasoning", "preference_following", "summarization", "temporal_reasoning"
    ]
    
    for category in category_order:
        if category in all_scores:
            scores = all_scores[category]
            avg = sum(scores) / len(scores)
            passed = sum(1 for s in scores if s >= 0.7)
            print(f"  {category:<30} {avg:.3f} (avg)  {passed}/{len(scores)} passed ({passed/len(scores)*100:.1f}%)")
    
    # Overall average across all categories
    all_category_scores = []
    for category in category_order:
        if category in all_scores:
            all_category_scores.extend(all_scores[category])
    
    overall_avg = sum(all_category_scores) / len(all_category_scores) if all_category_scores else 0
    total_passed = sum(1 for s in all_category_scores if s >= 0.7)
    
    print("-"*70)
    print(f"  {'OVERALL':<30} {overall_avg:.3f} (avg)  {total_passed}/{len(all_category_scores)} passed ({total_passed/len(all_category_scores)*100:.1f}%)")
    print("="*70)
    
    # Per-chat breakdown
    print("\n  📋 PER-CHAT BREAKDOWN:")
    print("-"*70)
    print(f"  {'Chat':<10} {'Overall':<10} {'Best':<25} {'Worst':<25}")
    print("-"*70)
    
    for chat in chat_scores_list:
        chat_id = chat["chat_id"]
        scores = chat["scores"]
        
        if "overall" in scores:
            overall = scores["overall"]
            
            # Find best and worst categories
            cat_scores = {k: v["average"] for k, v in scores.items() if k != "overall"}
            if cat_scores:
                best_cat = max(cat_scores, key=cat_scores.get)
                worst_cat = min(cat_scores, key=cat_scores.get)
                print(f"  Chat {chat_id:<5} {overall:.3f}        {best_cat:<25} {worst_cat:<25}")
    
    print("="*70)
    
    # Save detailed results
    output_file = f"results/beam_aggregated_results_{chat_size}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "chat_size": chat_size,
            "num_chats": len(chat_scores_list),
            "category_averages": {cat: sum(scores)/len(scores) for cat, scores in all_scores.items() if scores},
            "overall_average": overall_avg,
            "total_passed": total_passed,
            "total_questions": len(all_category_scores),
            "overall_accuracy": total_passed / len(all_category_scores) if all_category_scores else 0,
            "per_chat_results": chat_scores_list,
            "all_scores": dict(all_scores)
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    # Also save a simple CSV
    import csv
    csv_file = f"results/beam_aggregated_results_{chat_size}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Average Score", "Passed", "Total", "Accuracy"])
        
        for category in category_order:
            if category in all_scores:
                scores = all_scores[category]
                avg = sum(scores) / len(scores)
                passed = sum(1 for s in scores if s >= 0.7)
                total = len(scores)
                writer.writerow([category, f"{avg:.3f}", passed, total, f"{passed/total*100:.1f}%"])
        
        writer.writerow(["OVERALL", f"{overall_avg:.3f}", total_passed, len(all_category_scores), f"{total_passed/len(all_category_scores)*100:.1f}%"])
    
    print(f"💾 CSV saved to: {csv_file}")
    
    return chat_scores_list, all_scores

if __name__ == "__main__":
    aggregate_beam_results()