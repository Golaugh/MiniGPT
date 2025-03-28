import requests

def get_tokenizer(url):
    text = requests.get(url).text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    StoI = {ch: i for i, ch in enumerate(chars)}
    ItoS = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [StoI[c] for c in s]
    decode = lambda l: ''.join([ItoS[i] for i in l])
    return text, encode, decode, vocab_size