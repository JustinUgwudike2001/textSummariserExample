import numpy as np
import hnswlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from components import build_sbert_vectors

# 4a. VECTORISE AND L2-NORMALISE

def vectorise_sentences(sentences: list[str], bert = True) -> tuple[np.ndarray, object]:
    """
    Convert sentences to TF-IDF vectors and L2-normalise them.
 
    WHY L2-normalise?
    After normalisation every vector has magnitude = 1.
    This means: dot_product(a, b) == cosine_similarity(a, b)
    HNSW's inner product space ('ip') computes dot products, so this
    trick lets us do cosine similarity inside the HNSW index cheaply.
 
    Returns
    -------
    vectors    : np.ndarray shape (n, vocab_size), float32, unit-length
    vectorizer : fitted TfidfVectorizer (for inspection)

    """
    tfidf = None
    vectorizer = None

    if bert:
        tfidf, vectorizer = build_sbert_vectors(sentences)
    else:
        vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
        try:
            tfidf = vectorizer.fit_transform(sentences).toarray().astype('float32')
        except ValueError:
            # All words are stop words — return identity matrix as fallback
            n = len(sentences)
            return np.eye(n, dtype='float32'), vectorizer
 
    # L2-normalise: divide each row vector by its magnitude
    # Before: ||v|| could be anything
    # After:  ||v|| = 1.0 for every sentence
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1            # prevent divide-by-zero for blank sentences
    normalised = tfidf / norms
 
    return normalised, vectorizer


# 4b. BUILD THE HNSW INDEX

def build_hnsw_index(vectors: np.ndarray,
                     M: int = 16,
                     ef_construction: int = 100) -> hnswlib.Index:
    """
    Build an HNSW index from L2-normalised sentence vectors.
 
    Parameters
    ----------
    vectors         : unit-length vectors, shape (n, dim)
    M               : max bi-directional connections per node
                        8-16 = fast, lower memory usage
                        32-64 = more accurate, higher memory
    ef_construction : candidate list size during graph construction
                        100 is a safe, well-tested default
 
    Returns
    -------
    index : hnswlib.Index, ready to be queried
    """
    n, dim = vectors.shape
 
    # Initialise: inner product space on vectors of size `dim`
    index = hnswlib.Index(space='ip', dim=dim)
 
    # Allocate memory and build the layered graph
    index.init_index(
        max_elements=n,
        ef_construction=ef_construction,
        M=M
    )
 
    # Add all sentence vectors; IDs are their positions (0, 1, 2, ...)
    index.add_items(vectors, ids=list(range(n)))
 
    # Set query-time search width (can be adjusted after building)
    index.set_ef(max(50, M * 2))
 
    return index


# 4c. QUERY FOR NEAR-NEIGHBOURS

def find_near_duplicates_hnsw(vectors: np.ndarray,
                               index: hnswlib.Index,
                               threshold: float = 0.70,
                               k_neighbours: int = 5
                               ) -> list[tuple[int, int, float]]:
    """
    For each sentence, find its k nearest neighbours in the HNSW index
    and collect pairs whose similarity exceeds the threshold.
 
    Parameters
    ----------
    vectors      : L2-normalised sentence vectors
    index        : built HNSW index
    threshold    : cosine similarity cutoff — pairs above this are duplicates
                   For TF-IDF: 0.60-0.75 recommended
                   For SBERT : 0.85-0.92 recommended
    k_neighbours : how many neighbours to retrieve per sentence
 
    Returns
    -------
    duplicate_pairs : list of (i, j, similarity) tuples where i < j
    """
    n = len(vectors)
    duplicate_pairs = []
    seen_pairs = set()               # track (i,j) to avoid (j,i) duplicates
 
    for i in range(n):
        # knn_query returns ([labels], [distances]) each shape (1, k)
        labels, distances = index.knn_query(vectors[i], k=min(k_neighbours + 1, n))
        labels    = labels[0]        # flatten: shape (k,)
        distances = distances[0]     # flatten: shape (k,)
 
        for j, sim in zip(labels, distances):
            j   = int(j)
            sim = float(sim)
 
            if j == i:               # skip self-match
                continue
 
            pair = (min(i, j), max(i, j))
            if pair in seen_pairs:   # skip (j,i) if we saw (i,j)
                continue
            seen_pairs.add(pair)
 
            if sim >= threshold:
                duplicate_pairs.append((i, j, sim))
 
    return duplicate_pairs


def find_near_duplicates_exact(vectors: np.ndarray,
                                threshold: float = 0.70
                                ) -> list[tuple[int, int, float]]:
    """
    Brute-force O(N²) exact cosine similarity.
    Used for small documents (< ~50 sentences) or to verify HNSW results.
 
    Returns
    -------
    duplicate_pairs : list of (i, j, similarity) where i < j
    """
    sim_matrix = cosine_similarity(vectors)
    n = vectors.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim >= threshold:
                pairs.append((i, j, sim))
    return pairs


# 4d. RESOLVE DUPLICATES — decide which sentence to keep

def resolve_duplicates(sentences: list[str],
                        scores: np.ndarray,
                        duplicate_pairs: list[tuple[int, int, float]],
                        verbose: bool = True
                        ) -> set[int]:
    """
    Given detected duplicate pairs, remove the lower-scoring sentence.
 
    Strategy: always keep the sentence with the higher Step 3 score.
    Process most-similar pairs first so the most obvious duplicates
    are resolved before any cascading effects.
 
    Returns
    -------
    keep : set of indices to retain (all others are duplicates)
    """
    keep = set(range(len(sentences)))
 
    # Process most-similar pairs first
    for i, j, sim in sorted(duplicate_pairs, key=lambda x: -x[2]):
        if i not in keep or j not in keep:
            continue                 # already removed by a previous resolution
 
        # Keep the higher-scoring sentence
        if scores[i] >= scores[j]:
            keep.discard(j)
            winner, loser = i, j
        else:
            keep.discard(i)
            winner, loser = j, i
 
        if verbose:
            print(f"    Duplicate (sim={sim:.3f}): removed S{loser+1} "
                  f"(score={scores[loser]:.2f}), kept S{winner+1} "
                  f"(score={scores[winner]:.2f})")
 
    return keep


# FULL PIPELINE

def deduplicate(sentences: list[str],
                scores: np.ndarray,
                threshold: float = 0.70,
                hnsw_min_sentences: int = 20
                ) -> tuple[list[str], np.ndarray, list[int]]:
    """
    Full deduplication pipeline. Auto-selects exact vs HNSW based on N.
 
    Parameters
    ----------
    sentences           : cleaned sentences from Step 1
    scores              : combined scores from Step 3
    threshold           : cosine similarity cutoff (0.70 for TF-IDF)
    hnsw_min_sentences  : use HNSW when n >= this, exact otherwise
 
    Returns
    -------
    deduped_sentences : remaining sentences after deduplication
    deduped_scores    : their corresponding scores
    kept_indices      : original indices that survived
    """
    n = len(sentences)
    if n <= 1:
        return sentences, scores, list(range(n))
 
    # 4a: vectorise
    vectors, _ = vectorise_sentences(sentences)
 
    # 4b + 4c: find duplicates
    if n >= hnsw_min_sentences:
        print(f"  Method: HNSW  (n={n})")
        index = build_hnsw_index(vectors)
        pairs = find_near_duplicates_hnsw(vectors, index, threshold)
    else:
        print(f"  Method: Exact cosine  (n={n})")
        pairs = find_near_duplicates_exact(vectors, threshold)
 
    print(f"  Found {len(pairs)} duplicate pair(s) at threshold={threshold}")
 
    # 4d: resolve
    keep_set     = resolve_duplicates(sentences, scores, pairs)
    kept_indices = sorted(keep_set)
 
    # 4e: build output
    return (
        [sentences[i] for i in kept_indices],
        scores[np.array(kept_indices)],
        kept_indices
    )

