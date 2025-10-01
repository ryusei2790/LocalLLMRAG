# -*- coding: utf-8 -*-
import re
from typing import List

_SENT_SPLIT = re.compile(r"(?<=[。．！？\?\!])\s*|\n{2,}", re.MULTILINE)

def split_sentences(text: str) -> List[str]:
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    return sents

def greedy_chunk_by_tokens(
    text: str,
    tokenizer_like_len=lambda s: int(len(s) / 1.8),  # ざっくりトークン見積り
    target_tokens: int = 400,
    overlap_tokens: int = 60,
    min_chars: int = 150,
) -> List[str]:
    sents = split_sentences(text)
    chunks, cur, cur_toks = [], [], 0
    for sent in sents:
        stoks = tokenizer_like_len(sent)
        if cur_toks + stoks > target_tokens and cur:
            # 出力
            chunk = "".join(cur).strip()
            if len(chunk) >= min_chars:
                chunks.append(chunk)
            # オーバーラップ確保
            overlapped = []
            otoks = 0
            for s in reversed(cur):
                t = tokenizer_like_len(s)
                if otoks + t > overlap_tokens:
                    break
                overlapped.insert(0, s)
                otoks += t
            cur = overlapped + [sent]
            cur_toks = sum(tokenizer_like_len(x) for x in cur)
        else:
            cur.append(sent)
            cur_toks += stoks
    if cur:
        chunk = "".join(cur).strip()
        if len(chunk) >= min_chars:
            chunks.append(chunk)
    return chunks