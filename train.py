import argparse, json
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from data.tokenizer import get_tokenizer
from model.gpt import GPT
from transformers import get_linear_schedule_with_warmup
from utils.logging import StatsLogger

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        return SimpleNamespace(**json.load(f))

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

if __name__ == '__main__':
    cfg = parse_config()

    # Tokenizer
    text, encode, decode, vocab_size = get_tokenizer(cfg.URL)
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    # Model
    model = GPT(vocab_size, cfg.embed_dim, cfg.block_size, cfg.n_heads, cfg.ff_dim, cfg.n_layers).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=cfg.num_iters)
    logger = StatsLogger(log_dir="log")

    for step in range(cfg.num_iters):
        model.train()
        xb, yb = get_batch(train_data, cfg.block_size, cfg.batch_size, cfg.device)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_norm)
        optimizer.step()
        scheduler.step()

        if step % cfg.eval_interval == 0 or step == cfg.num_iters-1:
            stats = {
                "step": step,
                "train_loss": loss.item(),
                "lr": scheduler.get_last_lr()[0],
            }
            logger.log(stats)
            logger.log_console(stats)
            torch.save(model.state_dict(), "params/gpt.pt")

    print(f'Training completed!')