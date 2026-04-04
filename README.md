# BEAM Benchmark Evaluation (100K) - Results

## 🎯 Executive Summary

We evaluated GPT-4o on the BEAM benchmark's 100K token conversations, achieving **86.1% overall accuracy** across 20 chats and 374 probing questions.

## 📊 Results Summary (100K Chats - 20 chats, 374 questions)

| Category | Score | Questions | Verdict |
|----------|-------|-----------|---------|
| Abstention | **100%** | 40 | ✅ Perfect |
| Contradiction Resolution | **96.1%** | 38 | ✅ Excellent |
| Event Ordering | **76.3%** | 38 | ⚠️ Challenging |
| Information Extraction | **97.8%** | 38 | ✅ Excellent |
| Instruction Following | **50.0%** | 38 | ⚠️ Rubric-limited |
| Knowledge Update | **100%** | 38 | ✅ Perfect |
| Multi-Session Reasoning | **96.3%** | 36 | ✅ Excellent |
| Preference Following | **77.8%** | 36 | ⚠️ Moderate |
| Summarization | **100%** | 36 | ✅ Perfect |
| Temporal Reasoning | **65.3%** | 36 | ⚠️ Challenging |
| **OVERALL** | **86.1%** | **374** | 🏆 **Strong** |

## 📈 Per-Chat Performance

| Chat | Overall | Best Category | Worst Category |
|------|---------|---------------|----------------|
| 1 | 85.8% | Abstention | Instruction Following |
| 2 | 75.8% | Abstention | Instruction Following |
| 3 | 84.4% | Abstention | Event Ordering |
| 4 | 95.4% | Abstention | Event Ordering |
| 5 | **100%** | Abstention | - |
| 6 | 88.6% | Abstention | Instruction Following |
| 7 | 88.0% | Abstention | Instruction Following |
| 8 | 91.0% | Abstention | Event Ordering |
| 9 | 79.0% | Abstention | Preference Following |
| 10 | 81.0% | Abstention | Instruction Following |
| 11 | 90.0% | Abstention | Preference Following |
| 12 | 87.5% | Abstention | Instruction Following |
| 13 | 87.5% | Abstention | Instruction Following |
| 14 | 70.7% | Abstention | Instruction Following |
| 15 | 95.0% | Abstention | Instruction Following |
| 16 | 84.4% | Abstention | Instruction Following |
| 17 | 90.0% | Abstention | Instruction Following |
| 18 | 87.3% | Abstention | Instruction Following |
| 19 | 90.5% | Abstention | Instruction Following |
| 20 | 81.0% | Abstention | Event Ordering |

**Best Chat:** Chat 5 (100% overall)  
**Worst Chat:** Chat 14 (70.7% overall)

## 🔍 Key Findings

### Perfect Performance (3 categories)
- ✅ **Abstention** (100%) - Perfectly withholds answers when information is missing
- ✅ **Knowledge Update** (100%) - Correctly updates stored facts
- ✅ **Summarization** (100%) - Accurately compresses dialogue content

### Excellent Performance (3 categories)
- 📊 **Contradiction Resolution** (96.1%) - Detects inconsistencies across long contexts
- 📊 **Information Extraction** (97.8%) - Near-perfect recall of entities and facts
- 📊 **Multi-Session Reasoning** (96.3%) - Integrates evidence across non-adjacent turns

### Challenging Categories (4 categories)
- ⚠️ **Event Ordering** (76.3%) - Difficulty with long sequences (9+ items)
- ⚠️ **Instruction Following** (50.0%) - Rubrics demand specific formatting
- ⚠️ **Preference Following** (77.8%) - "Avoid" constraints are challenging
- ⚠️ **Temporal Reasoning** (65.3%) - Requires exact date range phrasing

## 📊 Category Difficulty Analysis

| Category | Score | Why? |
|----------|-------|------|
| Abstention | 100% | Clear negative constraints |
| Knowledge Update | 100% | Simple factual updates |
| Summarization | 100% | Well-defined task |
| Information Extraction | 97.8% | Direct factual recall |
| Contradiction Resolution | 96.1% | Clear logical inconsistencies |
| Multi-Session Reasoning | 96.3% | Strong cross-session integration |
| Preference Following | 77.8% | "Avoid" constraints are hard |
| Event Ordering | 76.3% | Long sequences (9+ items) |
| Temporal Reasoning | 65.3% | Exact phrasing requirements |
| Instruction Following | 50.0% | Rubric formatting demands |

## 📈 Performance by Chat
