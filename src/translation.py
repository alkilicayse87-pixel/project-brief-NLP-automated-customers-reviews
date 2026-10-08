"""English -> German review translation using Helsinki-NLP MarianMT models."""

from __future__ import annotations

import re
from functools import lru_cache

EN_DE_MODEL = "Helsinki-NLP/opus-mt-en-de"
DE_EN_MODEL = "Helsinki-NLP/opus-mt-de-en"
MAX_INPUT_CHARS = 2000

# Informal English / abbreviations that MarianMT often mishandles.
ABBREVIATIONS = {
    r"\bbtw\b": "by the way",
    r"\bimo\b": "in my opinion",
    r"\bimho\b": "in my opinion",
    r"\bidk\b": "I don't know",
    r"\bomg\b": "oh my god",
    r"\bfyi\b": "for your information",
    r"\bthx\b": "thanks",
    r"\bpls\b": "please",
    r"\bu\b": "you",
    r"\bur\b": "your",
    r"\bgr8\b": "great",
    r"\bw/o\b": "without",
    r"\bw/\b": "with",
    r"\bcuz\b": "because",
    r"\bgonna\b": "going to",
    r"\bwanna\b": "want to",
    r"\bkinda\b": "kind of",
    r"\bdidnt\b": "didn't",
    r"\bdont\b": "don't",
    r"\bcant\b": "can't",
    r"\bdoesnt\b": "doesn't",
    r"\bwont\b": "won't",
    r"\bisnt\b": "isn't",
    r"\bive\b": "I've",
}

# Product names that must stay untranslated (placeholders survive Marian).
PROTECTED_TERMS = [
    "Fire Tablet", "Fire HD", "Kindle Paperwhite", "Kindle Fire", "Kindle", "Echo Dot",
    "Echo", "Alexa", "Amazon", "Fire TV", "Fire Stick", "Prime", "Whispersync",
    "Paperwhite", "Fire",
]


def normalize_informal(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([!?.])\1{2,}", r"\1", text)
    text = re.sub(r"(\w)\1{3,}", r"\1\1", text)  # soooo -> soo
    # Sentence-initial "Great tablet" is read as the name "Große"; make it a noun phrase.
    text = re.sub(
        r"(^|[.!?]\s+)great\s+(?!(?:for|and|but|to|with|on|in|at|as|if|so|when|it|that|than)\b)(?=\w)",
        lambda m: f"{m.group(1)}An awesome ",
        text,
        flags=re.IGNORECASE,
    )
    for pattern, replacement in ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def detect_language(text: str) -> str:
    """Return an ISO code such as 'en' or 'de', or 'unknown'."""
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0
    if len(re.findall(r"[A-Za-zÄÖÜäöüß]", text)) < 3:
        return "unknown"
    try:
        return detect(text)
    except Exception:
        return "unknown"


@lru_cache(maxsize=2)
def _load(model_name: str):
    from transformers import MarianMTModel, MarianTokenizer

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p.strip()]


def _protect(text: str):
    """Swap product names for placeholders; each occurrence is restored verbatim."""
    mapping: dict[str, str] = {}
    for term in sorted(PROTECTED_TERMS, key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(term)}(?:s)?\b", re.IGNORECASE)

        def swap(match: re.Match) -> str:
            token = f"PRD{len(mapping)}X"
            mapping[token] = match.group(0)
            return token

        text = pattern.sub(swap, text)
    return text, mapping


# MarianMT renders "tablet" as the pharmaceutical "Tablette"; in German tech usage it is "Tablet".
GERMAN_FIXES = {
    r"\bTabletten\b": "Tablets",
    r"\bTablette\b": "Tablet",
    r"\bTablettes\b": "Tablets",
}

# Words ending in "e" that are determiners: they lose the "e" before a neuter noun.
_DETERMINERS = ("eine", "diese", "keine", "meine", "deine", "seine", "ihre", "unsere", "welche", "jede", "solche")


def _fix_tablet_gender(text: str) -> str:
    """Tablet is neuter in German; correct feminine endings the model produces."""

    def repl(match: re.Match) -> str:
        word = match.group(1)
        if word.lower() in _DETERMINERS:
            ending = "es" if word.lower() in ("diese", "welche", "jede", "solche") else ""
            return f"{word[:-1]}{ending} Tablet"
        return f"{word}s Tablet"

    return re.sub(r"\b(\w+e) Tablet\b", repl, text)



def _restore(text: str, mapping: dict[str, str]) -> str:
    for token, term in mapping.items():
        text = re.sub(re.escape(token), term, text, flags=re.IGNORECASE)
    for pattern, replacement in GERMAN_FIXES.items():
        text = re.sub(pattern, replacement, text)
    text = _fix_tablet_gender(text)
    return re.sub(r"\b([Ee])ine (\w+es Tablet)", r"\1in \2", text)


def _run(model_name: str, sentences: list[str], batch_size: int = 8) -> list[str]:
    import torch

    tokenizer, model = _load(model_name)
    outputs: list[str] = []
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start:start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            generated = model.generate(**encoded, num_beams=4, max_new_tokens=256)
        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return outputs


def translate_en_to_de(text: str) -> dict:
    """Translate one review. Returns dict with status, language, and translation."""
    text = (text or "").strip()
    if not text:
        return {"status": "empty", "language": "unknown", "translation": ""}
    truncated = len(text) > MAX_INPUT_CHARS
    text = text[:MAX_INPUT_CHARS]
    language = detect_language(text)
    if language == "de":
        return {"status": "already_german", "language": "de", "translation": text}
    if language not in ("en", "unknown"):
        return {"status": "unsupported_language", "language": language, "translation": ""}

    prepared, mapping = _protect(normalize_informal(text))
    sentences = _split_sentences(prepared)
    translated = _run(EN_DE_MODEL, sentences)
    result = _restore(" ".join(translated), mapping)
    return {
        "status": "translated_truncated" if truncated else "translated",
        "language": language,
        "translation": result,
    }


def back_translate_de_to_en(text: str) -> str:
    if not text.strip():
        return ""
    return " ".join(_run(DE_EN_MODEL, _split_sentences(text)))
