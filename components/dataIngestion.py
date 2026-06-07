import re
import unicodedata
import nltk
from nltk.tokenize import sent_tokenize
 
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

# 1a. INGESTION — load text from different sources

def ingest_from_string(text: str) -> str:
    """Accept raw text directly."""
    return text
 
 
def ingest_from_file(filepath: str) -> str:
    """Read plain text from a .txt file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()
 
 
def ingest_from_url(url: str) -> str:
    """
    Fetch article text from a URL.
    Requires: pip install requests beautifulsoup4
    """
    import requests
    from bs4 import BeautifulSoup
 
    response = requests.get(url, timeout=10)
    response.raise_for_status()
 
    soup = BeautifulSoup(response.text, 'html.parser')
 
    # Remove script and style blocks
    for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
        tag.decompose()
 
    return soup.get_text(separator=' ')
#____________________________________________________________
# 1b. NOISE REMOVAL — strip things that aren't real content

def remove_html_tags(text: str) -> str:
    """
    Strip any HTML/XML tags.
    e.g. '<p>Hello <b>world</b></p>'  →  'Hello world'
    """
    return re.sub(r'<[^>]+>', ' ', text)
 
 
def remove_urls(text: str) -> str:
    """
    Remove http links and www addresses.
    e.g. 'Visit https://example.com for more'  →  'Visit  for more'
    """
    return re.sub(r'http\S+|www\.\S+', '', text)
 
 
def remove_emails(text: str) -> str:
    """
    Remove email addresses.
    e.g. 'Contact us at info@example.com'  →  'Contact us at'
    """
    return re.sub(r'\S+@\S+\.\S+', '', text)
 
 
def remove_excessive_punctuation(text: str) -> str:
    """
    Collapse repeated punctuation down to a single character.
    e.g. 'Wow!!!'  →  'Wow!'
         'Hmm...'  →  'Hmm.'
    """
    return re.sub(r'([!?.]){2,}', r'\1', text)
 
 
def remove_control_characters(text: str) -> str:
    """
    Remove invisible/non-printable control characters.
    These often sneak in from copy-pasted text or web scraping.
    """
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

def remove_whitespaces(text: str) -> str:
    text = re.sub(r'\n', '', text)
    return text.strip().replace('\r', '')
 
 
def remove_noise(text: str) -> str:
    """Run all noise removal steps in sequence."""
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_excessive_punctuation(text)
    text = remove_control_characters(text)
    #text = remove_whitespaces(text)
    return text
#____________________________________________________________
# 1c. NORMALISATION — standardise format and encoding

def normalise_unicode(text: str) -> str:
    """
    Normalise unicode so accented/special characters are consistent.
    NFKC: converts characters like ﬁ (ligature) → fi (two chars)
    """
    return unicodedata.normalize('NFKC', text)
 
 
def normalise_quotes(text: str) -> str:
    """
    Convert curly/smart quotes to straight quotes.
    e.g. "hello" → "hello"   'world' → 'world'
    """
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    return text
 
 
def normalise_dashes(text: str) -> str:
    """
    Convert em-dashes and en-dashes to regular hyphens.
    e.g. 'state–of–the–art'  →  'state-of-the-art'
    """
    return re.sub(r'[–—]', '-', text)
 
 
def normalise_whitespace(text: str) -> str:
    """
    Collapse multiple spaces, tabs, and newlines into a single space.
    e.g. 'hello   \n\n  world'  →  'hello world'
    """
    return re.sub(r'\s+', ' ', text).strip()
 
 
def normalise(text: str) -> str:
    """Run all normalisation steps in sequence."""
    text = normalise_unicode(text)
    text = normalise_quotes(text)
    text = normalise_dashes(text)
    text = normalise_whitespace(text)
    return text
#____________________________________________________________
# 1d. SENTENCE SEGMENTATION — split text into individual sentences

def segment_sentences(text: str) -> list[str]:
    """
    Use NLTK's Punkt tokenizer to split text into sentences.
    It handles abbreviations like 'Dr.', 'U.S.A.', etc. correctly.
 
    e.g. 'Hello world. How are you?'
         → ['Hello world.', 'How are you?']
    """
    sentences = sent_tokenize(text)
    # Strip whitespace from each sentence
    return [s.strip() for s in sentences if s.strip()]
#____________________________________________________________
# 1e. QUALITY GATE — filter out sentences that aren't useful

def is_too_short(sentence: str, min_words: int = 5) -> bool:
    """
    Reject sentences with fewer than min_words words.
    e.g. 'Ok.' or 'See figure 3.' are not useful for summarisation.
    """
    return len(sentence.split()) < min_words
 
 
def is_too_long(sentence: str, max_words: int = 100) -> bool:
    """
    Reject sentences longer than max_words.
    Very long sentences are often run-ons, table rows, or legal boilerplate.
    """
    return len(sentence.split()) > max_words
 
 
def is_mostly_non_alpha(sentence: str, threshold: float = 0.5) -> bool:
    """
    Reject sentences where less than 50% of characters are letters.
    Catches things like: '| 2.3 | 4.5 | 6.7 |' (table data)
    """
    if not sentence:
        return True
    alpha_count = sum(c.isalpha() for c in sentence)
    return (alpha_count / len(sentence)) < threshold
 
 
def is_repetitive(sentence: str, repeat_threshold: float = 0.5) -> bool:
    """
    Reject sentences where a single word makes up more than 50% of the text.
    Catches things like: 'the the the the the the the'
    """
    words = sentence.lower().split()
    if not words:
        return True
    most_common_count = max(words.count(w) for w in set(words))
    return (most_common_count / len(words)) > repeat_threshold
 
 
def quality_gate(sentences: list[str],
                 min_words: int = 5,
                 max_words: int = 100) -> tuple[list[str], list[str]]:
    """
    Run all quality checks. Returns (kept, rejected) sentence lists.
    """
    kept     = []
    rejected = []
 
    for sent in sentences:
        if is_too_short(sent, min_words):
            rejected.append((sent, 'too short'))
        elif is_too_long(sent, max_words):
            rejected.append((sent, 'too long'))
        elif is_mostly_non_alpha(sent):
            rejected.append((sent, 'mostly non-alpha'))
        elif is_repetitive(sent):
            rejected.append((sent, 'repetitive'))
        else:
            kept.append(sent)
 
    return kept, rejected