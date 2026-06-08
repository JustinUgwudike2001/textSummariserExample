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

    scores = get_scores(quality_text)
    print(scores)

    features = combine_features(
        sentences=quality_text,
        textrank_scores_arr=scores['textrank'],
        centroid_scores_arr=scores['centroid'],
        domain='general'
    )
    print(features)
