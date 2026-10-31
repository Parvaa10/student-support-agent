"""Tiny tokenizer + light stemmer shared by policy retrieval and memory selection (no extra dependencies)."""
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_WORD = re.compile(r"[a-z0-9&]+")
_EXTRA_STOP = {"pgdm", "student", "students", "can", "i", "my", "me", "please", "tell"}


def stem(word: str) -> str:
    """Very small suffix-stripper so 'failed', 'fails', 'failure' -> 'fail'."""
    if len(word) <= 3:
        return word
    for suf, rep in (("ies", "y"), ("ure", ""), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            word = word[: -len(suf)] + rep
            break
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    words = _WORD.findall(text.lower())
    return [stem(w) for w in words if w not in ENGLISH_STOP_WORDS and w not in _EXTRA_STOP]
