from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]{2,}")


def tokenize(text: str) -> List[str]:
	if not text:
		return []
	return _TOKEN_PATTERN.findall(text.lower())


def normalize_scores(scores: Sequence[float]) -> np.ndarray:
	arr = np.asarray(scores, dtype=np.float32)
	if arr.size == 0:
		return arr
	max_val = float(np.max(arr))
	min_val = float(np.min(arr))
	if max_val - min_val <= 1e-8:
		if max_val <= 1e-8:
			return np.zeros_like(arr)
		return np.ones_like(arr)
	return (arr - min_val) / (max_val - min_val)


@dataclass
class BM25Index:
	doc_len: List[int]
	term_freqs: List[Dict[str, int]]
	idf: Dict[str, float]
	avgdl: float
	k1: float = 1.5
	b: float = 0.75

	@classmethod
	def from_documents(cls, docs_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> "BM25Index":
		term_freqs: List[Dict[str, int]] = []
		doc_len: List[int] = []
		doc_freq: Dict[str, int] = {}

		for tokens in docs_tokens:
			tf: Dict[str, int] = {}
			for token in tokens:
				tf[token] = tf.get(token, 0) + 1
			term_freqs.append(tf)
			doc_len.append(len(tokens))
			for token in tf.keys():
				doc_freq[token] = doc_freq.get(token, 0) + 1

		n_docs = max(1, len(docs_tokens))
		avgdl = float(sum(doc_len)) / n_docs if doc_len else 0.0
		idf = {
			token: math.log(((n_docs - freq + 0.5) / (freq + 0.5)) + 1.0)
			for token, freq in doc_freq.items()
		}

		return cls(doc_len=doc_len, term_freqs=term_freqs, idf=idf, avgdl=avgdl, k1=k1, b=b)

	def get_scores(self, query_tokens: Sequence[str]) -> np.ndarray:
		if not self.term_freqs:
			return np.zeros(0, dtype=np.float32)
		if not query_tokens:
			return np.zeros(len(self.term_freqs), dtype=np.float32)

		scores = np.zeros(len(self.term_freqs), dtype=np.float32)
		for idx, tf in enumerate(self.term_freqs):
			dl = self.doc_len[idx] if idx < len(self.doc_len) else 0
			denom_base = self.k1 * (1 - self.b + self.b * (dl / self.avgdl if self.avgdl > 0 else 0.0))
			score = 0.0
			for token in query_tokens:
				if token not in tf:
					continue
				freq = tf[token]
				idf = self.idf.get(token, 0.0)
				num = freq * (self.k1 + 1.0)
				denom = freq + denom_base
				score += idf * (num / denom if denom > 0 else 0.0)
			scores[idx] = score
		return scores

	def top_n(self, query_tokens: Sequence[str], n: int) -> List[tuple[int, float]]:
		scores = self.get_scores(query_tokens)
		if scores.size == 0:
			return []
		n = max(1, min(n, scores.size))
		top_idx = np.argsort(scores)[::-1][:n]
		return [(int(i), float(scores[i])) for i in top_idx]
