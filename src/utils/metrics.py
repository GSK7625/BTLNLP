# -*- coding: utf-8 -*-
import re
import string
from collections import Counter

def normalize_answer(s: str) -> str:
    """Chuẩn hóa: lowercase, bỏ dấu câu, bỏ article, collapse khoảng trắng."""
    if not s:
        return ""
        
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def get_tokens(s: str) -> list[str]:
    return normalize_answer(s).split()


def compute_exact(gold: str, pred: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(pred))


def compute_f1(gold: str, pred: str) -> float:
    gold_toks = get_tokens(gold)
    pred_toks = get_tokens(pred)
    if not gold_toks or not pred_toks:
        return 0.0
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    f1 = 2 * precision * recall / (precision + recall)
    return f1
