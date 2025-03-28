import torch
from model.gpt import GPT
from data.tokenizer import get_tokenizer
from train import parse_config

def sample(model, x, max_new_tokens):
    for _ in range(max_new_tokens):
        logits = model(x)
        logits = logits[:, -1, :]  # take last token
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_token), dim=1)
    return x

if __name__ == '__main__':
    cfg = parse_config()
    _, encode, decode, vocab_size = get_tokenizer(cfg.URL)

    model = GPT(vocab_size, cfg.embed_dim, cfg.block_size, cfg.n_heads, cfg.ff_dim, cfg.n_layers)
    model.load_state_dict(torch.load("params/gpt.pt"))
    model.eval()

    start_text = "Those who realize"
    x = torch.tensor([encode(start_text)], dtype=torch.long).to(cfg.device)
    y = sample(model.to(cfg.device), x, max_new_tokens=cfg.max_new_tokens)
    print(decode(y[0].tolist()))