import json
import os
import math
import re
from collections import defaultdict

class SimpleBM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        """
        corpus: list of lists of tokens, e.g. [["this","is","doc1"], ["another","doc"]]
        """
        self.corpus = corpus
        self.N = len(corpus)
        self.k1 = k1
        self.b = b

        self.doc_freq = {}   # term -> in how many docs
        self.doc_len = []    # len(doc_i)
        self.avgdl = 0.0

        for doc in corpus:
            self.doc_len.append(len(doc))
            seen = set()
            for term in doc:
                if term not in seen:
                    self.doc_freq[term] = self.doc_freq.get(term, 0) + 1
                    seen.add(term)

        self.avgdl = sum(self.doc_len) / max(self.N, 1)

    def idf(self, term: str) -> float:
        n = self.doc_freq.get(term, 0)
        if n == 0:
            return 0.0
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def score(self, query_tokens, index: int) -> float:
        doc = self.corpus[index]
        freqs = {}
        for t in doc:
            freqs[t] = freqs.get(t, 0) + 1

        score = 0.0
        dl = self.doc_len[index]
        for term in query_tokens:
            if term not in freqs:
                continue
            tf = freqs[term]
            idf = self.idf(term)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * (tf * (self.k1 + 1) / denom)
        return score

    def get_scores(self, query_tokens):
        return [self.score(query_tokens, i) for i in range(self.N)]


def tokenize(text: str):
    return re.findall(r"\w+", text.lower())

def getBestPassage(file_obj):
    passages = []
    first_obj = None

    for line in file_obj:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)

        if first_obj is None:
            first_obj = obj 

        passages.append({
            "passage_id": obj["passage_id"],
            "text": obj["passage"],
            "search_rank": obj["search_rank"],
            "url": obj["url"],
        })

    if not passages or first_obj is None:
        return None

    # BM25 over passages for this query
    tokenized_passages = [tokenize(p["text"]) for p in passages]
    tokenized_query = tokenize(first_obj["query"])

    bm25 = SimpleBM25(tokenized_passages)
    scores = bm25.get_scores(tokenized_query)

    # best passage index
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_passage = passages[best_idx]

    result = {
        "query_id": first_obj["query_id"],
        "query": first_obj["query"],
        "best_passage_id": best_passage["passage_id"],
        "best_passage": best_passage["text"],
        "bm25_score": float(scores[best_idx]),
        "search_rank": best_passage["search_rank"],
        "url": best_passage["url"],
    }

    return result

for subfolder in os.listdir("passages"):
    subfolder_path = os.path.join("passages", subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    print("Processing doc:", subfolder)

    for filename in os.listdir(subfolder_path):
        file_path = os.path.join(subfolder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            best = getBestPassage(f)
            if not best:
                print(f"ERROR ON {subfolder}/{filename}")
            with open(f"bestPassages/{subfolder}.jsonl", "a", encoding="utf-8") as out:
                out.write(json.dumps(best, ensure_ascii=False) + '\n')


