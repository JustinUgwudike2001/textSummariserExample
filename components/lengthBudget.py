import numpy as np

#5a COMPRESSION RATIO BUDGET

def ratio_budget(n_sentences: int,
                 compression_ratio: float = 0.3,
                 min_sentences: int = 1,
                 max_sentences: int = 20) -> int:
    """
    Compute how many sentences to keep based on a compression ratio.
 
    Parameters
    ----------
    n_sentences      : total number of sentences after deduplication
    compression_ratio: fraction of sentences to keep (0.0 to 1.0)
                         0.1 = very aggressive (keep 10%)
                         0.3 = moderate (keep 30%) — good default
                         0.5 = light (keep 50%)
    min_sentences    : always keep at least this many (floor)
    max_sentences    : never keep more than this many (ceiling)
 
    Returns
    -------
    budget : int
 
    Examples
    --------
    20 sentences, ratio=0.30 → keep 6
    50 sentences, ratio=0.20 → keep 10
     5 sentences, ratio=0.30 → keep 2 (rounds up from 1.5, then hits min=1)
    """
    raw    = n_sentences * compression_ratio
    budget = max(min_sentences, round(raw))
    budget = min(budget, max_sentences)
    budget = min(budget, n_sentences)   # can't keep more than we have
    return budget


# 5b. WORD COUNT BUDGET

def word_count_budget(sentences: list[str],
                      scores: np.ndarray,
                      max_words: int = 150) -> int:
    """
    Compute how many sentences fit within a word count ceiling.
 
    Strategy: greedily add sentences in descending score order,
    counting words until the next sentence would exceed max_words.
 
    Parameters
    ----------
    sentences : deduplicated sentences from Step 4
    scores    : their combined scores from Step 3
    max_words : hard word count limit for the whole summary
 
    Returns
    -------
    budget : int — number of sentences that fit within max_words
 
    Examples
    --------
    max_words=150, sentences averaging 20 words → budget ≈ 7
    max_words=50,  sentences averaging 20 words → budget ≈ 2
    """
    # Sort by score descending — best sentences get priority
    sorted_idx = np.argsort(scores)[::-1]
 
    total_words = 0
    budget      = 0
 
    for idx in sorted_idx:
        wc = len(sentences[idx].split())
 
        # Would adding this sentence exceed the word limit?
        if total_words + wc > max_words:
            break                  # stop — next sentence doesn't fit
 
        total_words += wc
        budget      += 1
 
    return max(1, budget)          # always allow at least 1 sentence


# 5c. SCORE THRESHOLD BUDGET

def score_threshold_budget(scores: np.ndarray,
                            threshold: float = 0.25) -> int:
    """
    Count how many sentences score above the minimum quality threshold.
 
    Any sentence below the threshold is considered too low-quality to
    include, regardless of what the ratio or word count budget allows.
 
    Parameters
    ----------
    scores    : combined scores from Step 3, shape (n,)
    threshold : minimum score to qualify (0.0 to 1.0)
                  0.10 = very permissive (almost everything qualifies)
                  0.25 = moderate — good default
                  0.50 = strict (only clearly important sentences)
 
    Returns
    -------
    budget : int — number of sentences above the threshold
 
    Examples
    --------
    scores=[0.9, 0.7, 0.5, 0.3, 0.1], threshold=0.25
        → 4 sentences qualify (0.9, 0.7, 0.5, 0.3 are all >= 0.25)
    """
    qualifying = int(np.sum(scores >= threshold))
    return max(1, qualifying)      # always allow at least 1


# 5d. COMBINE ALL CONSTRAINTS

def compute_budget(sentences: list[str],
                   scores: np.ndarray,
                   compression_ratio: float = 0.3,
                   max_words: int = 150,
                   score_threshold: float = 0.25,
                   min_sentences: int = 1,
                   max_sentences: int = 20,
                   verbose: bool = True) -> dict:
    """
    Run all three constraints and return the most restrictive budget.
 
    Parameters
    ----------
    sentences         : deduplicated sentences from Step 4
    scores            : their combined scores from Step 3
    compression_ratio : fraction of sentences to keep
    max_words         : hard word count ceiling for the summary
    score_threshold   : minimum score to qualify for inclusion
    min_sentences     : absolute floor on budget
    max_sentences     : absolute ceiling on budget
    verbose           : print breakdown of each constraint
 
    Returns
    -------
    dict with:
        budget          : int  — final number of sentences to select  ← use this
        ratio_b         : int  — ratio-based budget
        word_count_b    : int  — word-count-based budget
        threshold_b     : int  — score-threshold-based budget
        binding         : str  — which constraint was most restrictive
        qualifying_scores: np.ndarray — mask of sentences above threshold
    """
    n = len(sentences)
 
    # Run each constraint independently
    ratio_b      = ratio_budget(n, compression_ratio, min_sentences, max_sentences)
    word_count_b = word_count_budget(sentences, scores, max_words)
    threshold_b  = score_threshold_budget(scores, score_threshold)
 
    # Most restrictive wins
    final = min(ratio_b, word_count_b, threshold_b)
    final = max(final, min_sentences)   # never go below the floor
 
    # Identify which constraint is binding (most restrictive)
    budgets = {
        "compression_ratio": ratio_b,
        "word_count":        word_count_b,
        "score_threshold":   threshold_b,
    }
    binding = min(budgets, key=budgets.get)
 
    # Which sentences individually qualify by score
    qualifying_mask = scores >= score_threshold
 
    if verbose:
        total_words = sum(len(s.split()) for s in sentences)
        print(f"  Input          : {n} sentences, {total_words} total words")
        print(f"  Ratio budget   : {ratio_b}  "
              f"({compression_ratio:.0%} of {n} sentences)")
        print(f"  Word-count bgt : {word_count_b}  "
              f"(best sentences within {max_words} words)")
        print(f"  Threshold bgt  : {threshold_b}  "
              f"({int(qualifying_mask.sum())} sentences ≥ {score_threshold} score)")
        print(f"  ─────────────────────────────")
        print(f"  Final budget   : {final}  ← binding constraint: {binding}")
 
    return {
        "budget":            final,
        "ratio_b":           ratio_b,
        "word_count_b":      word_count_b,
        "threshold_b":       threshold_b,
        "binding":           binding,
        "qualifying_mask":   qualifying_mask,
    }


# HELPER: show how budget changes across settings

def budget_sensitivity(sentences: list[str],
                        scores: np.ndarray) -> None:
    """
    Print a table showing how each parameter affects the final budget.
    Useful for tuning before you run the full pipeline.
    """
    print("\n── Sensitivity: Compression Ratio ──")
    print(f"  {'Ratio':>8}  {'Budget':>8}  {'Est. words':>12}")
    print("  " + "-" * 32)
    for ratio in [0.10, 0.20, 0.30, 0.40, 0.50]:
        b = ratio_budget(len(sentences), ratio)
        # Estimate words: take top-b sentences by score, sum their words
        top_idx   = np.argsort(scores)[::-1][:b]
        est_words = sum(len(sentences[i].split()) for i in top_idx)
        print(f"  {ratio:>8.0%}  {b:>8}  {est_words:>12}")
 
    print("\n── Sensitivity: Max Words ──")
    print(f"  {'Max words':>10}  {'Budget':>8}")
    print("  " + "-" * 22)
    for mw in [50, 75, 100, 150, 200, 300]:
        b = word_count_budget(sentences, scores, max_words=mw)
        print(f"  {mw:>10}  {b:>8}")
 
    print("\n── Sensitivity: Score Threshold ──")
    print(f"  {'Threshold':>10}  {'Qualifying':>12}  {'Sentences above'}") 
    print("  " + "-" * 55)
    for t in [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]:
        b    = score_threshold_budget(scores, threshold=t)
        qual = [f"S{i+1}({scores[i]:.2f})" for i in range(len(scores))
                if scores[i] >= t]
        print(f"  {t:>10.2f}  {b:>12}  {', '.join(qual)}")