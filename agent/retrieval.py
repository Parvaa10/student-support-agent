"""Policy retrieval (RAG) over the student handbook.

* The handbook is split into one chunk per numbered sub-section ("### 2.3 Payment portal failure").
* Chunks are indexed with TF-IDF (unigrams + bigrams, light stemming) and ranked by cosine similarity.
* Only the top-k chunks above a minimum score are returned, so the model never sees the whole document.
"""
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .text_utils import tokenize


@dataclass
class Chunk:
    id: str        # e.g. "POL-2.3"
    title: str     # e.g. "Fees > Payment portal failure"
    text: str


def load_chunks(path: Path) -> list[Chunk]:
    chunks, section, cur = [], "", None
    for line in path.read_text(encoding="utf-8").splitlines():
        h2 = re.match(r"^##\s+\d+\.\s+(.*)", line)
        h3 = re.match(r"^###\s+(\d+\.\d+)\s+(.*)", line)
        if h2:
            section = h2.group(1).strip()
        elif h3:
            if cur:
                chunks.append(cur)
            cur = Chunk(id=f"POL-{h3.group(1)}", title=f"{section} > {h3.group(2).strip()}", text="")
        elif cur and line.strip():
            cur.text += line.strip() + " "
    if cur:
        chunks.append(cur)
    for c in chunks:
        c.text = c.text.strip()
    return chunks


def _analyzer(text: str) -> list[str]:
    toks = tokenize(text)
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


class PolicyRetriever:
    def __init__(self, path: Path = config.HANDBOOK_PATH,
                 top_k: int = config.RETRIEVAL_TOP_K, min_score: float = config.RETRIEVAL_MIN_SCORE):
        self.chunks = load_chunks(path)
        self.top_k, self.min_score = top_k, min_score
        self.vectorizer = TfidfVectorizer(analyzer=_analyzer, sublinear_tf=True)
        # Title is repeated so headings weigh a little more than body text.
        self.matrix = self.vectorizer.fit_transform([f"{c.title} {c.title} {c.text}" for c in self.chunks])

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self.top_k
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        out = []
        for i in sims.argsort()[::-1][:k]:
            if sims[i] >= self.min_score:
                out.append({**asdict(self.chunks[i]), "score": round(float(sims[i]), 3)})
        return out
