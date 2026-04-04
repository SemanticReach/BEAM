import json
from pathlib import Path
from collections import defaultdict

def aggregate_beam_results(base_directory="eval_structured", chat_size="100K", num_chats=20):
    """
    Aggregate BEAM evaluation results from all chat directories.
    
    Args:
        base_directory: Directory containing numbered subfolders (1/, 2/, etc.)
        chat_size: Size of chats (100K, 500K, etc.)
        num_chats: Number of chats to process (default 20 for 100K)
    """
    
    all_scores = defaultdict(list)
    detailed_results = []
    
    for i in range(1, num_chats + 1):
        eval_file = Path(base_directory) / str(i) / f"evaluation-answers_gpt4o_{chat_size}_{i}.json"
        
        if not eval_file.exists():
            print(f"Warning: {eval_file} not found")
            continue
        
        with open(eval_file, 'r') as f:
            data = json.load(f)
        
        chat_scores = {}
        
        for category, items in data.items():
            category_scores = []
            for item in items:
                if "llm_judge_score" in item:
                    score = item["llm_judge_score"]
                    category_scores.append(score)
            
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
        
        detailed_results.append({
            "chat_id": i,
            "scores": chat_scores
        })
    
    # Print results
    print("\n" + "="*70)
    print(f"BEAM BENCHMARK RESULTS - {chat_size} CHATS ({len(detailed_results)} chats)")
    print("="*70)
    
    # Category averages
    print("\n📊 PER-CATEGORY RESULTS:")
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
            print(f"  {category:<30} {avg:.3f} ({len(scores)} questions)")
    
    # Overall average across all categories
    all_category_scores = []
    for category in category_order:
        if category in all_scores:
            all_category_scores.extend(all_scores[category])
    
    overall_avg = sum(all_category_scores) / len(all_category_scores) if all_category_scores else 0
    print("-"*70)
    print(f"  {'OVERALL AVERAGE':<30} {overall_avg:.3f} ({len(all_category_scores)} questions)")
    print("="*70)
    
    # Per-chat breakdown
    print("\n📋 PER-CHAT RESULTS:")
    print("-"*70)
    print(f"  {'Chat':<10} {'Overall':<10} {'Best Category':<20} {'Worst Category':<20}")
    print("-"*70)
    
    for chat in detailed_results:
        chat_id = chat["chat_id"]
        scores = chat["scores"]
        
        if "overall" in scores:
            overall = scores["overall"]
            
            # Find best and worst categories
            cat_scores = {k: v["average"] for k, v in scores.items() if k != "overall"}
            if cat_scores:
                best_cat = max(cat_scores, key=cat_scores.get)
                worst_cat = min(cat_scores, key=cat_scores.get)
                print(f"  Chat {chat_id:<5} {overall:.3f}        {best_cat:<20} {worst_cat:<20}")
    
    print("="*70)
    
    # Save detailed results to file
    output_file = f"beam_aggregated_results_{chat_size}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "chat_size": chat_size,
            "num_chats": len(detailed_results),
            "category_averages": {cat: sum(scores)/len(scores) for cat, scores in all_scores.items()},
            "overall_average": overall_avg,
            "per_chat_results": detailed_results
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    return detailed_results, all_scores

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate BEAM evaluation results")
    parser.add_argument("--input_dir", type=str, default="eval_structured", 
                        help="Directory containing numbered subfolders with results")
    parser.add_argument("--chat_size", type=str, default="100K",
                        help="Chat size (100K, 500K, 1M, 10M)")
    parser.add_argument("--num_chats", type=int, default=20,
                        help="Number of chats to process")
    
    args = parser.parse_args()
    
    aggregate_beam_results(
        base_directory=args.input_dir,
        chat_size=args.chat_size,
        num_chats=args.num_chats
    )