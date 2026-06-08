from components import *

if __name__ == '__main__':
    
    # step 1 - ingestion
    # raw_text = ingest_from_url("https://jamesclear.com/saying-no")
    # filtered_text = remove_noise(raw_text)
    # normalised_text = normalise(filtered_text)
    # segmented_text = segment_sentences(normalised_text)
    # quality_text = quality_gate(segmented_text)

    quality_text = ingest_using_url("https://jamesclear.com/saying-no")
    print(quality_text)

    # step 2 - multi-signal scoring

    scores = get_scores(quality_text)
    print(scores)

    # step 3 - feature extraction

    features = combine_features(
        sentences=quality_text,
        textrank_scores_arr=scores['textrank'],
        centroid_scores_arr=scores['centroid'],
        domain='general'
    )
    print(features)

    # step 4 - deduplication

    THRESHOLD   = 0.60
    deduped_sents, deduped_scores, kept_idx = deduplicate(
    quality_text, features['combined'], threshold=THRESHOLD, hnsw_min_sentences=20
    )

    print(f'KEPT IDs={kept_idx}')

    # step 5 - length budget

    budget_information = compute_budget(
        quality_text, features['combined'],
        compression_ratio=0.30,
        max_words=150,
        score_threshold=0.25,
    )
    budget = budget_information["budget"]
    print(budget)


