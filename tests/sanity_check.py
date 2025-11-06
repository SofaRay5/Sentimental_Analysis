from src.preprocess import clean_text, tokenize, generate_ngrams, build_vocab

texts = [
    "I love this movie, it's awesome!",
    "Not good at all, I hated it."
]
docs_tokens = [generate_ngrams(tokenize(clean_text(s)), True) for s in texts]
print(docs_tokens)
vocab = build_vocab(docs_tokens, min_freq=1, max_size=50)
print(vocab)
