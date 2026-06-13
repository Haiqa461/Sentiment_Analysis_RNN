"""
demo.py
-------
Quick manual test: trains a small sentiment model for a few epochs,
then lets you type tweets and see live predictions.

Run:
    python demo.py
"""

import re
import time
import torch
import torch.nn as nn
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from custom_lstm_cell import CustomLSTMCell, SentimentLSTM

# ── config ────────────────────────────────────────────────────────────────────
CSV_PATH   = "dataset/training.1600000.processed.noemoticon.csv"
N_ROWS     = 20_000    # 10k neg + 10k pos  — fast to train
MAX_LEN    = 30
MIN_FREQ   = 2
BATCH_SIZE = 128
EPOCHS     = 5
LR         = 1e-3
EMBED_DIM  = 64
HIDDEN     = 128
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── regex patterns ────────────────────────────────────────────────────────────
_URL  = re.compile(r"https?://\S+|www\.\S+")
_MEN  = re.compile(r"@\w+")
_HASH = re.compile(r"#(\w+)")
_PUNC = re.compile(r"[^\w\s]")
_SPC  = re.compile(r"\s+")


def clean(text):
    text = _URL.sub("", text)
    text = _MEN.sub("", text)
    text = _HASH.sub(r"\1", text)
    text = text.lower()
    text = _PUNC.sub(" ", text)
    return _SPC.sub(" ", text).strip()


# ── vocabulary ────────────────────────────────────────────────────────────────
PAD, UNK = "<PAD>", "<UNK>"

class Vocab:
    def __init__(self):
        self.w2i = {PAD: 0, UNK: 1}
        self.size = 2

    def build(self, token_lists, min_freq=2):
        cnt = Counter(w for toks in token_lists for w in toks)
        for w, f in cnt.most_common():
            if f < min_freq:
                break
            if w not in self.w2i:
                self.w2i[w] = self.size
                self.size += 1

    def encode_pad(self, tokens, max_len):
        ids = [self.w2i.get(w, 1) for w in tokens][:max_len]
        ids += [0] * (max_len - len(ids))
        return ids


# ── dataset ───────────────────────────────────────────────────────────────────
class TweetDS(Dataset):
    def __init__(self, ids, labels):
        self.ids    = ids
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):  return len(self.labels)

    def __getitem__(self, i):
        return torch.tensor(self.ids[i], dtype=torch.long), self.labels[i]


# ── train / eval helpers ──────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = total_correct = n = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE).unsqueeze(1)
            logits = model(xb)
            loss   = criterion(logits, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (preds == yb).sum().item()
            total_loss    += loss.item() * yb.size(0)
            n             += yb.size(0)

    return total_loss / n, total_correct / n


# ── predict on raw text ───────────────────────────────────────────────────────
def predict(model, vocab, text):
    tokens = clean(text).split()
    ids    = vocab.encode_pad(tokens, MAX_LEN)
    x      = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()
    label = "POSITIVE" if prob >= 0.5 else "NEGATIVE"
    bar_len = int(prob * 30)
    bar = "#" * bar_len + "-" * (30 - bar_len)
    return prob, label, bar


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("=" * 60)
    print("  Sentiment Analysis — Manual Demo")
    print(f"  Device : {DEVICE}")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    cols = ["label", "id", "date", "flag", "user", "text"]
    half = N_ROWS // 2
    neg = pd.read_csv(CSV_PATH, encoding="latin-1", header=None,
                      names=cols, skiprows=0, nrows=half)
    pos = pd.read_csv(CSV_PATH, encoding="latin-1", header=None,
                      names=cols, skiprows=800_000, nrows=half)
    df = pd.concat([neg, pos], ignore_index=True)
    df["label"] = df["label"].map({0: 0, 4: 1})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df["text"] = df["text"].astype(str).apply(clean)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    print(f"    Loaded {len(df):,} tweets  "
          f"(neg={int((df['label']==0).sum()):,}  pos={int((df['label']==1).sum()):,})")

    # ── 2. Split + vocab ──────────────────────────────────────────────────────
    print("\n[2/5] Splitting and building vocabulary...")
    train_df, test_df = train_test_split(df, test_size=0.2,
                                         stratify=df["label"], random_state=42)
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    train_df["tokens"] = train_df["text"].apply(str.split)
    test_df["tokens"]  = test_df["text"].apply(str.split)

    vocab = Vocab()
    vocab.build(train_df["tokens"], min_freq=MIN_FREQ)
    print(f"    Train={len(train_df):,}  Test={len(test_df):,}  "
          f"Vocab={vocab.size:,}")

    # ── 3. Encode + DataLoaders ───────────────────────────────────────────────
    print("\n[3/5] Encoding tweets...")
    train_ids = [vocab.encode_pad(t, MAX_LEN) for t in train_df["tokens"]]
    test_ids  = [vocab.encode_pad(t, MAX_LEN) for t in test_df["tokens"]]

    train_loader = DataLoader(TweetDS(train_ids, train_df["label"].tolist()),
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(TweetDS(test_ids,  test_df["label"].tolist()),
                              batch_size=BATCH_SIZE, shuffle=False)

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print(f"\n[4/5] Training for {EPOCHS} epochs...")
    model     = SentimentLSTM(vocab_size=vocab.size,
                              embed_dim=EMBED_DIM,
                              hidden_size=HIDDEN).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"    {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Test Acc':>8}  {'Time':>5}")
    print("    " + "-" * 45)

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        _,       te_acc = run_epoch(model, test_loader,  criterion)
        print(f"    {ep:>5}  {tr_loss:>10.4f}  {tr_acc*100:>8.2f}%  "
              f"{te_acc*100:>7.2f}%  {time.time()-t0:>4.1f}s")

    # ── 5. Interactive predictions ────────────────────────────────────────────
    print("\n[5/5] Testing on sample tweets...")
    samples = [
        "I absolutely love this! Best day ever, feeling amazing!",
        "This is terrible, I hate everything about today. So frustrated.",
        "Just had coffee. It was fine.",
        "Can't believe how wonderful my friends are, so grateful!",
        "My laptop crashed and I lost all my work. Worst day.",
        "The weather is okay I guess",
    ]

    print()
    print("  " + "=" * 56)
    print("  Sample Predictions")
    print("  " + "=" * 56)
    for tweet in samples:
        prob, label, bar = predict(model, vocab, tweet)
        print(f"\n  Tweet : {tweet[:55]}")
        print(f"  [{bar}] {prob:.2%}  ->  {label}")

    print()
    print("  " + "=" * 56)
    print("  Type your own tweet (or 'quit' to exit)")
    print("  " + "=" * 56)

    while True:
        try:
            user_input = input("\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("quit", "exit", "q", ""):
            break
        prob, label, bar = predict(model, vocab, user_input)
        print(f"  [{bar}] {prob:.2%}  ->  {label}")

    print("\n  Done.")
