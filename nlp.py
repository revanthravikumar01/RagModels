"""
NLP utilities for the Farmers Assistant Chatbot.

Everything related to processing human language (apart from the neural
models themselves) lives here, so it is easy to read and explain:

  1. Text normalisation   - fix PDF symbols, Unicode forms and spacing
  2. Tokenisation         - split text into lowercase word tokens
  3. Stopword removal     - drop common words like "the", "how", "is"
  4. Stemming             - reduce word forms ("varieties" -> "variety")
  5. Query expansion      - expand farming abbreviations ("BPH" -> "brown planthopper")
  6. Follow-up handling   - resolve "it"/"this" using the previous question
  7. BM25 keyword search  - classic information-retrieval ranking
  8. TF-IDF keywords      - find the most important words of a document
  9. Text statistics      - word / sentence counts for the insights page
 10. Highlighting         - mark question words inside the source text
"""

import html
import math
import re
import unicodedata
from collections import Counter

import numpy as np


# ---------------------------------------------------------------------------
# 1. Text normalisation
# ---------------------------------------------------------------------------

# Phrases that are written in more than one way in agricultural documents.
# Both the documents and the questions are rewritten to the same form.
SPELLING_VARIANTS = {
    r"\bplant\s*hoppers?\b": "planthopper",
    r"\bleaf\s*hoppers?\b": "leafhopper",
    r"\bleaf\s*folders?\b": "leaffolder",
    r"\bleaf\s*rollers?\b": "leaffolder",
    r"\bearhead\s*bugs?\b": "earhead bug",
    r"\bfertili[sz]ers?\b": "fertilizer",
}


def normalize_text(text: str) -> str:
    """Clean raw PDF text so that it is readable and consistent.

    - NFKC turns special forms into normal characters (e.g. the 'ﬁ' ligature -> 'fi')
    - PDF bullet symbols from private Unicode areas become '•'
    - Repeated spaces are collapsed and empty lines removed
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\uE000-\uF8FF]", "•", text)  # private-use glyphs (PDF bullets)
    text = text.replace("\u00a0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


# ---------------------------------------------------------------------------
# 2-4. Tokenisation, stopwords and stemming
# ---------------------------------------------------------------------------

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been
before being below between both but by can could did do does doing down during
each few for from further had has have having he her here hers herself him
himself his i if in into is it its itself just me more most my myself no nor
not now of off on once only or other our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there these
they this those through to too under until up very was we were what when where
which while who whom why will with would you your yours yourself yourselves
how tell give explain please kindly want know need using use used also may
shall must get got one two us let lets
""".split())

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def stem(word: str) -> str:
    """A small, safe stemmer for plurals ("hoppers" -> "hopper").

    Full stemmers (like Porter) cut words aggressively ("disease" -> "diseas"),
    which can hurt chemical and variety names, so only plurals are handled.
    """
    if len(word) <= 3 or word.isdigit():
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"           # varieties -> variety
    if word.endswith(("sses", "shes", "ches", "xes")):
        return word[:-2]                 # classes -> class, patches -> patch
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]                 # hoppers -> hopper
    return word


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Lowercase, apply spelling variants, split into words, drop stopwords, stem."""
    text = text.lower()
    for pattern, replacement in SPELLING_VARIANTS.items():
        text = re.sub(pattern, replacement, text)
    tokens = TOKEN_PATTERN.findall(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [stem(t) for t in tokens]


# ---------------------------------------------------------------------------
# 5. Query expansion: farming abbreviations
# ---------------------------------------------------------------------------

ABBREVIATIONS = {
    "bph": "brown planthopper",
    "wbph": "white backed planthopper",
    "glh": "green leafhopper",
    "blb": "bacterial leaf blight",
    "rtv": "rice tungro virus",
    "etl": "economic threshold level",
    "ipm": "integrated pest management",
    "inm": "integrated nutrient management",
    "iwm": "integrated weed management",
    "dsr": "direct seeded rice",
    "sri": "system of rice intensification",
    "fym": "farmyard manure",
    "psb": "phosphate solubilizing bacteria",
    "dat": "days after transplanting",
    "das": "days after sowing",
    "npk": "nitrogen phosphorus potassium",
    "tnau": "Tamil Nadu Agricultural University",
    "kvk": "Krishi Vigyan Kendra",
    "ec": "emulsifiable concentrate",
    "wp": "wettable powder",
    "sc": "suspension concentrate",
    "wg": "water dispersible granules",
}


def expand_abbreviations(question: str) -> tuple[str, list[str]]:
    """Add the full form after each known abbreviation.

    "How to control BPH?" -> "How to control BPH (brown planthopper)?"
    Returns the expanded question and the list of expansions made.
    """
    expansions = []

    def replace(match):
        word = match.group(0)
        full = ABBREVIATIONS[word.lower()]
        expansions.append(f"{word} = {full}")
        return f"{word} ({full})"

    pattern = r"\b(" + "|".join(ABBREVIATIONS) + r")\b"
    expanded = re.sub(pattern, replace, question, flags=re.IGNORECASE)
    return expanded, expansions


# ---------------------------------------------------------------------------
# 6. Follow-up questions ("How do I control it?")
# ---------------------------------------------------------------------------

FOLLOW_UP_WORDS = {"it", "its", "this", "that", "these", "those", "they", "them", "same"}


def is_follow_up(question: str) -> bool:
    """A question is treated as a follow-up if it is short, refers back with a
    pronoun, and has very few content words of its own."""
    words = re.findall(r"[a-z]+", question.lower())
    content_words = tokenize(question)
    return (any(w in FOLLOW_UP_WORDS for w in words)
            and len(content_words) <= 3)


def build_search_query(question: str, previous_question: str | None = None) -> dict:
    """Prepare the text that is actually searched for, and explain what changed."""
    notes = []
    query = question.strip()

    if previous_question and is_follow_up(query):
        query = f"{previous_question.strip()} {query}"
        notes.append("Follow-up question: combined with your previous question")

    query, expansions = expand_abbreviations(query)
    if expansions:
        notes.append("Expanded abbreviations: " + ", ".join(expansions))

    return {"query": query, "keywords": tokenize(query), "notes": notes}


# ---------------------------------------------------------------------------
# 7. BM25 keyword search
# ---------------------------------------------------------------------------

class BM25Index:
    """Okapi BM25: the classic keyword ranking used by search engines.

    A chunk scores high when it contains the question's words often (term
    frequency), when those words are rare in the whole collection (inverse
    document frequency), and it is adjusted for chunk length.
    """

    def __init__(self, tokenized_chunks: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(tokenized_chunks)
        self.doc_lengths = np.array([len(t) for t in tokenized_chunks], dtype=np.float32)
        self.avg_length = float(self.doc_lengths.mean()) if self.n_docs else 0.0

        # postings[word] = list of (chunk index, how often the word appears)
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for index, tokens in enumerate(tokenized_chunks):
            for word, count in Counter(tokens).items():
                self.postings.setdefault(word, []).append((index, count))

        self.idf = {
            word: math.log(1 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5))
            for word, p in self.postings.items()
        }

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        if not self.n_docs:
            return scores
        for word in set(query_tokens):
            for index, freq in self.postings.get(word, []):
                length_norm = 1 - self.b + self.b * self.doc_lengths[index] / self.avg_length
                scores[index] += self.idf[word] * freq * (self.k1 + 1) / (freq + self.k1 * length_norm)
        return scores


# ---------------------------------------------------------------------------
# 8. TF-IDF keywords for a document
# ---------------------------------------------------------------------------

# Words that are frequent in agricultural reports but say nothing about the
# topic (job titles in author lists, units).
KEYWORD_EXCLUDE = {"prof", "professor", "assoc", "associate", "asst", "assistant",
                   "head", "dept", "scientist", "kg", "ha", "ml", "day"}


def top_keywords(doc_token_lists: list[list[str]], idf: dict[str, float], n: int = 10) -> list[str]:
    """Most important words of one document: frequent in it (TF) and not
    common everywhere in the knowledge base (IDF)."""
    counts = Counter(t for tokens in doc_token_lists for t in tokens
                     if len(t) >= 3 and not t[0].isdigit() and t not in KEYWORD_EXCLUDE)
    scored = {word: count * idf.get(word, 0.0) for word, count in counts.items()}
    return [word for word, _ in sorted(scored.items(), key=lambda x: -x[1])[:n]]


# ---------------------------------------------------------------------------
# 9. Text statistics
# ---------------------------------------------------------------------------

def text_statistics(text: str) -> dict:
    words = re.findall(r"[A-Za-z]+", text)
    sentences = [s for s in re.split(r"[.!?]+\s|\n{2,}", text) if len(s.split()) >= 3]
    return {
        "words": len(words),
        "unique_words": len({w.lower() for w in words}),
        "sentences": len(sentences),
        "avg_sentence_length": round(len(words) / max(len(sentences), 1), 1),
    }


# ---------------------------------------------------------------------------
# 10. Highlighting question words in source text
# ---------------------------------------------------------------------------

def highlight_html(text: str, keywords: list[str]) -> str:
    """Return safe HTML with the question's keywords wrapped in <mark>.

    The PDF text is HTML-escaped first, so it can never inject HTML or
    scripts into the page.
    """
    safe = html.escape(text)
    # Use a word stem so "variety" also highlights "varieties".
    stems = {k[:-1] if len(k) > 5 else k for k in keywords if len(k) >= 3}
    words = sorted(stems, key=len, reverse=True)
    if words:
        pattern = r"\b(" + "|".join(re.escape(w) for w in words) + r")\w*"
        safe = re.sub(pattern, r"<mark>\g<0></mark>", safe, flags=re.IGNORECASE)
    return safe.replace("\n", "<br>")
