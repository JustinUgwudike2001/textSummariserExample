from components import *

if __name__ == '__main__':

    # step 1 - ingestion
    raw_text = ingest_from_url("https://jamesclear.com/saying-no")
    filtered_text = remove_noise(raw_text)
    normalised_text = normalise(filtered_text)
    segmented_text = segment_sentences(normalised_text)
    quality_text = quality_gate(segmented_text)

    print(quality_text)
