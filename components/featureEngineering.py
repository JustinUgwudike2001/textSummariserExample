import re
import math
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
 
nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

# 3a. POSITION SCORE

def position_scores(sentences: list[str]) -> np.ndarray:
    """
    Score each sentence by its position in the document.
 
    Returns
    -------
    scores : np.ndarray of shape (n,), normalised to [0, 1]
    """
    n = len(sentences)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])
 
    scores = []
    for i in range(n):
        # relative = 0.0 at first sentence, 1.0 at last sentence
        relative = i / (n - 1)
 
        if relative <= 0.5:
            # First half: score decreases from 1.0 → 0.5
            score = 1.0 - relative
        else:
            # Second half: slight uptick at the end (conclusion)
            score = 0.3 + 0.3 * (1.0 - relative)
 
        scores.append(score)
 
    scores = np.array(scores)
 
    # Normalise to [0, 1]
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        scores = (scores - s_min) / (s_max - s_min)
 
    return scores

# 3b. KEYWORD / ENTITY OVERLAP SCORE

def extract_keywords(text: str, top_n: int = 15) -> set[str]:
    """
    Extract the top N most frequent meaningful words from the full document.
 
    Parameters
    ----------
    text  : the full document as a single string
    top_n : how many keywords to extract
 
    Returns
    -------
    keywords : set of keyword strings (lowercased)
    """
    stop_words = set(stopwords.words('english'))
 
    # Tokenise and clean
    tokens = word_tokenize(text.lower())
    tokens = [
        t for t in tokens
        if t.isalpha()           # letters only (no punctuation, numbers)
        and t not in stop_words  # remove stop words
        and len(t) > 3           # skip very short words like 'war', 'big'
    ]
 
    # Count and take top N
    freq = Counter(tokens)
    keywords = {word for word, _ in freq.most_common(top_n)}
    return keywords
 
 
def keyword_overlap_scores(sentences: list[str],
                            keywords: set[str]) -> np.ndarray:
    """
    Score each sentence by the fraction of document keywords it contains.
 
    Score = (keywords found in sentence) / (total keywords)
 
    A sentence mentioning 8 out of 15 keywords scores 8/15 = 0.53
 
    Returns
    -------
    scores : np.ndarray of shape (n,), normalised to [0, 1]
    """
    if not sentences or not keywords:
        return np.zeros(len(sentences))
 
    scores = []
    for sent in sentences:
        # Tokenise the sentence into a set of unique words
        tokens = set(word_tokenize(sent.lower()))
        # Count how many document keywords appear in this sentence
        overlap = len(tokens & keywords)
        scores.append(overlap / len(keywords))
 
    scores = np.array(scores, dtype=float)
 
    # Normalise to [0, 1]
    s_max = scores.max()
    if s_max > 0:
        scores = scores / s_max
 
    return scores

# 3c. LENGTH PENALTY SCORE

def length_penalty_scores(sentences: list[str],
                           ideal_min: int = 10,
                           ideal_max: int = 35) -> np.ndarray:
    """
    Score sentences by how close their word count is to the ideal range.
 
    Parameters
    ----------
    ideal_min : minimum ideal word count (below this = penalised)
    ideal_max : maximum ideal word count (above this = penalised)
 
    Returns
    -------
    scores : np.ndarray of shape (n,), values in [0, 1]
 
    Examples
    --------
    5 words   → 5/10 = 0.50  (too short)
    15 words  → 1.00          (ideal)
    30 words  → 1.00          (ideal)
    50 words  → ~0.72         (a bit long)
    100 words → ~0.52         (too long)
    """
    scores = []
    for sent in sentences:
        wc = len(sent.split())
 
        if ideal_min <= wc <= ideal_max:
            score = 1.0
 
        elif wc < ideal_min:
            # Linear scale up from 0 to 1 as we approach ideal_min
            score = wc / ideal_min
 
        else:
            # Logarithmic penalty for sentences longer than ideal
            # log1p(0) = 0 → no penalty at ideal_max
            # log1p(65) ≈ 4.2 → heavy penalty at 100 words
            excess = wc - ideal_max
            score = max(0.0, 1.0 - 0.05 * math.log1p(excess))
 
        scores.append(score)
 
    return np.array(scores)

# 3d. DOMAIN WEIGHT PRESETS

DOMAIN_WEIGHTS = {
    "general": {
        "textrank": 0.30,
        "centroid": 0.25,
        "position": 0.15,
        "keyword":  0.20,
        "length":   0.10,
    },
    "news": {
        # Lead bias: position matters most
        "textrank": 0.25,
        "centroid": 0.20,
        "position": 0.30,
        "keyword":  0.20,
        "length":   0.05,
    },
    "academic": {
        # Structure and topic coverage are key
        "textrank": 0.35,
        "centroid": 0.30,
        "position": 0.10,
        "keyword":  0.20,
        "length":   0.05,
    },
    "legal": {
        # Definitions and named entities matter most
        "textrank": 0.25,
        "centroid": 0.25,
        "position": 0.10,
        "keyword":  0.35,
        "length":   0.05,
    },
}

# 3e. SCORE COMBINATION

def combine_features(sentences: list[str],
                     textrank_scores_arr: np.ndarray,
                     centroid_scores_arr: np.ndarray,
                     domain: str = "general") -> dict:
    """
    Combine all 5 signals into a single final score per sentence.
 
    Parameters
    ----------
    sentences           : list of cleaned sentence strings (from Step 1)
    textrank_scores_arr : TextRank scores from Step 2
    centroid_scores_arr : centroid scores from Step 2
    domain              : "general" | "news" | "academic" | "legal"
 
    Returns
    -------
    dict containing:
        textrank  : TextRank signal
        centroid  : centroid signal
        position  : position signal
        keyword   : keyword overlap signal
        length    : length penalty signal
        combined  : final weighted score  ← pass this to Step 4
        keywords  : the keyword set used
        domain    : domain used
        weights   : weights used
    """
    weights   = DOMAIN_WEIGHTS.get(domain, DOMAIN_WEIGHTS["general"])
    full_text = " ".join(sentences)
 
    # Compute the 3 new features
    pos_scores = position_scores(sentences)
    keywords   = extract_keywords(full_text, top_n=15)
    kw_scores  = keyword_overlap_scores(sentences, keywords)
    len_scores = length_penalty_scores(sentences)
 
    # Weighted combination
    combined = (
        weights["textrank"] * textrank_scores_arr +
        weights["centroid"] * centroid_scores_arr +
        weights["position"] * pos_scores +
        weights["keyword"]  * kw_scores +
        weights["length"]   * len_scores
    )
 
    # Re-normalise combined to [0, 1]
    c_min, c_max = combined.min(), combined.max()
    if c_max > c_min:
        combined = (combined - c_min) / (c_max - c_min)
 
    return {
        "textrank": textrank_scores_arr,
        "centroid": centroid_scores_arr,
        "position": pos_scores,
        "keyword":  kw_scores,
        "length":   len_scores,
        "combined": combined,
        "keywords": keywords,
        "domain":   domain,
        "weights":  weights,
    }