import re
from collections import Counter

def clean_text(s: str) -> str:
    s = s.lower()                      # 1. 全部小写
    s = re.sub(r"<br\s*/?>", " ", s)   # 2. 去掉 HTML 标签
    s = re.sub(r"[^a-z0-9\s]", " ", s) # 3. 去掉非字母数字
    s = re.sub(r"\s+", " ", s).strip() # 4. 合并多余空格
    return s

def tokenize(s: str) -> list[str]:
    return s.split()

def generate_ngrams(tokens: list[str], use_bigram: bool = True) -> list[str]:
    if not use_bigram or len(tokens) < 2:
        return tokens
    bigrams = [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]
    return tokens + bigrams


def build_vocab(docs_tokens, min_freq=2, max_size=20000):
    counter = Counter()
    for tokens in docs_tokens:
        counter.update(tokens)

    # 过滤低频词
    items = [(w, c) for w, c in counter.items() if c >= min_freq]

    # 按频次降序排列
    items.sort(key=lambda x: (-x[1], x[0]))

    # 截断
    items = items[:max_size]

    vocab = {"<UNK>": 0}
    for i, (w, _) in enumerate(items, start=1):
        vocab[w] = i
    return vocab

