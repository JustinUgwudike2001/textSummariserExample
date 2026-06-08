import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# BACKGROUND: What is TF-IDF?

def build_tfidf_vectors(sentences: list[str]) -> tuple[np.ndarray, object]:
    """
    Convert sentences to TF-IDF vectors.
 
    Returns
    -------
    matrix     : shape (n_sentences, vocab_size)  — one row per sentence
    vectorizer : fitted TfidfVectorizer (kept for later use if needed)
    """
    vectorizer = TfidfVectorizer(
        stop_words='english',   # ignore 'the', 'is', 'and', etc.
        min_df=1,               # include words that appear at least once
        max_df=0.95,            # ignore words in >95% of sentences (too common)
    )
    matrix = vectorizer.fit_transform(sentences).toarray()
    return matrix.astype('float32'), vectorizer

# 2a. TEXTRANK

def build_similarity_matrix(sentences: list[str], method) -> np.ndarray:
    """
    Build an N×N matrix where entry [i,j] = cosine similarity between
    sentence i and sentence j.
 
    This is the adjacency matrix / edge weights of the TextRank graph.
    Self-similarity (diagonal) is set to 0 so a sentence doesn't vote for itself.
 
    Example for 3 sentences:
         S1    S2    S3
    S1 [ 0.0   0.8   0.2 ]
    S2 [ 0.8   0.0   0.3 ]
    S3 [ 0.2   0.3   0.0 ]
    """
    if len(sentences) < 2:
        return np.ones((len(sentences), len(sentences)))
 
    matrix, _ = method(sentences)
    sim = cosine_similarity(matrix)
    np.fill_diagonal(sim, 0.0)   # remove self-loops
    return sim
 
 
def textrank_scores(sentences: list[str],
                    embedd_method,
                    damping: float = 0.85,
                    max_iterations: int = 100,
                    tolerance: float = 1e-4) -> np.ndarray:
    """
    Run the TextRank power iteration algorithm.
 
    Parameters
    ----------
    damping        : probability of following a graph edge (vs. random jump)
                     higher = more influence from graph structure
    max_iterations : stop after this many rounds even if not converged
    tolerance      : stop early if scores change by less than this
 
    Returns
    -------
    scores : np.ndarray of shape (n_sentences,), normalised to [0, 1]
    """
    n = len(sentences)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])
 
    # Step 1: build the similarity graph
    sim_matrix = build_similarity_matrix(sentences, embedd_method)
 
    # Step 2: row-normalise → transition probabilities
    # Each row sums to 1, so sim_matrix[i] = probability of moving FROM i
    row_sums = sim_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1              # avoid divide-by-zero for isolated nodes
    transition = sim_matrix / row_sums
 
    # Step 3: initialise all scores equally
    scores = np.ones(n) / n
 
    # Step 4: power iteration
    for iteration in range(max_iterations):
        # Each sentence's new score = random jump + weighted sum of neighbours' scores
        # transition.T @ scores  →  how much "vote" flows INTO each sentence
        new_scores = (1 - damping) / n + damping * (transition.T @ scores)
 
        # Check convergence — stop if scores barely changed
        delta = np.linalg.norm(new_scores - scores)
        scores = new_scores
 
        if delta < tolerance:
            print(f"    TextRank converged at iteration {iteration + 1}")
            break
 
    # Step 5: normalise final scores to [0, 1]
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        scores = (scores - s_min) / (s_max - s_min)
 
    return scores

# 2b. CENTROID SIMILARITY

def centroid_scores(sentences: list[str]) -> np.ndarray:
    """
    Score each sentence by cosine similarity to the document centroid.
 
    A high score means: "this sentence captures the overall topic well."
    A low score means: "this sentence is off-topic or tangential."
 
    Returns
    -------
    scores : np.ndarray of shape (n_sentences,), normalised to [0, 1]
    """
    if not sentences:
        return np.array([])
    if len(sentences) == 1:
        return np.array([1.0])
 
    # Step 1: vectorise
    matrix, _ = build_tfidf_vectors(sentences)
 
    # Step 2: compute centroid — mean of all sentence vectors
    centroid = matrix.mean(axis=0, keepdims=True)   # shape: (1, vocab_size)
 
    # Step 3: cosine similarity between each sentence and the centroid
    # cosine_similarity returns shape (n_sentences, 1) — flatten to 1D
    scores = cosine_similarity(matrix, centroid).flatten()
 
    # Step 4: normalise to [0, 1]
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        scores = (scores - s_min) / (s_max - s_min)
 
    return scores

# 2c. COMBINE BOTH SIGNALS

def multi_signal_scores(sentences: list[str],
                        embed_method,
                        textrank_weight: float = 0.5,
                        centroid_weight: float = 0.5) -> dict:
    """
    Run both scoring methods and combine them into one score per sentence.
 
    Parameters
    ----------
    textrank_weight : how much to weight the TextRank score  (default 0.5)
    centroid_weight : how much to weight the centroid score  (default 0.5)
                      weights don't have to sum to 1, but it's cleaner if they do
 
    Returns
    -------
    dict with keys:
        textrank  : raw TextRank scores
        centroid  : raw centroid scores
        combined  : weighted combination of both
    """
    print("  Running TextRank scoring...")
    tr_scores  = textrank_scores(sentences, embed_method)
 
    print("  Running centroid similarity scoring...")
    cen_scores = centroid_scores(sentences)
 
    combined = textrank_weight * tr_scores + centroid_weight * cen_scores
 
    # Re-normalise combined score to [0, 1]
    c_min, c_max = combined.min(), combined.max()
    if c_max > c_min:
        combined = (combined - c_min) / (c_max - c_min)
 
    return {
        "textrank":  tr_scores,
        "centroid":  cen_scores,
        "combined":  combined,
    }

# To use SBERT, install: pip install sentence-transformers
# Then replace build_tfidf_vectors() with this:
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # small, fast model

def build_sbert_vectors(sentences: list[str]) -> tuple[np.ndarray, object]:
    return (model.encode(sentences, convert_to_numpy=True), None)

def get_scores(sentences: list[str], bert = True) -> dict:
    if bert:
        return multi_signal_scores(sentences, embed_method=build_sbert_vectors)
    else:
        return multi_signal_scores(sentences, embed_method=build_tfidf_vectors)
