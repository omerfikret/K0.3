import time
import sys
import os
import zipfile
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer

# --- MAIN.PY İLE BİREBİR EŞİTLENMİŞ PARAMETRELER (42.2M) ---
block_size = 512
n_embd = 512
n_head = 8
n_layer = 8
dropout = 0.10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- YEREL DOSYA YOLLARI ---
TOKENIZER_FILE = 'tokenizer_16k.json'
BEST_MODEL_PATH = 'K0.3.1/42M_model1_st200M.pth'

if not os.path.exists(TOKENIZER_FILE):
    raise FileNotFoundError(f"Tokenizer bulunamadı: {TOKENIZER_FILE}")
if not os.path.exists(BEST_MODEL_PATH):
    raise FileNotFoundError(f"Eğitilmiş model bulunamadı: {BEST_MODEL_PATH}")

tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
vocab_size = tokenizer.get_vocab_size()
encode = lambda s: tokenizer.encode(s).ids
decode = lambda l: tokenizer.decode(l)

# Özel token'ları üretim esnasında engelleme
_SPECIAL_TOKEN_CANDIDATES = [
    "<unk>", "[UNK]", "<pad>", "[PAD]", "<s>", "</s>",
    "[BOS]", "[EOS]", "<|endoftext|>", "[CLS]", "[SEP]", "[MASK]",
]
_vocab = tokenizer.get_vocab()
BANNED_TOKEN_IDS = sorted({_vocab[t] for t in _SPECIAL_TOKEN_CANDIDATES if t in _vocab})

# ================== MODEL MİMARİSİ ==================
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout_p = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out))

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, dropout)
        self.ffwd = FeedFoward(n_embd)
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.ln_f(self.blocks(tok_emb + pos_emb))
        return self.lm_head(x), None

    @torch.no_grad()
    def complete_text(self, prompt_text, max_new_tokens=150, temperature=0.75,
                      top_k=40, top_p=0.9, repetition_penalty=1.2, stop_at_sentence=True):
        """
        Düz metin tamamlama (GPT-2 stili). 
        stop_at_sentence=True yapıldığında son yarım kalan cümleyi temizler.
        """
        prompt_ids = encode(prompt_text)
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated_ids = []

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # 1. Yasaklı token'ları engelle
            if BANNED_TOKEN_IDS:
                logits[0, BANNED_TOKEN_IDS] = -float('Inf')

            # 2. Repetition Penalty
            if repetition_penalty is not None and repetition_penalty != 1.0:
                for prev_token in set(idx[0].tolist()):
                    if logits[0, prev_token] < 0:
                        logits[0, prev_token] *= repetition_penalty
                    else:
                        logits[0, prev_token] /= repetition_penalty

            # 3. Top-K Filtreleme
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # 4. Top-P (Nucleus) Filtreleme
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False

                indices_to_remove = sorted_indices[0][sorted_indices_to_remove[0]]
                logits[0, indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            token_id = idx_next.item()
            generated_ids.append(token_id)
            idx = torch.cat((idx, idx_next), dim=1)

        # Ham üretilen metin
        full_generated_text = decode(generated_ids)

        # Cümle Sonu Temizleme Mantığı (Nokta, Ünlem, Soru İşareti)
        if stop_at_sentence:
            end_marks = ['.', '!', '?']
            last_mark_idx = max(full_generated_text.rfind(mark) for mark in end_marks)
            
            # Eğer üretilen metinde en az bir nokta/cümle sonu varsa oraya kadar kes
            if last_mark_idx != -1:
                cleaned_text = full_generated_text[:last_mark_idx + 1]
            else:
                cleaned_text = full_generated_text
        else:
            cleaned_text = full_generated_text

        # Ekrana temizlenmiş çıktıyı bas
        sys.stdout.write(cleaned_text)
        sys.stdout.flush()

        print("\n")
        return cleaned_text

def resolve_pth_path(path):
    if os.path.isfile(path):
        return path

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Ne dosya ne klasör bulunamadı: {path}")

    real_root = None
    for root, _, files in os.walk(path):
        if 'data.pkl' in files:
            real_root = root
            break

    if real_root is None:
        raise FileNotFoundError(f"'{path}' klasörünün içinde 'data.pkl' bulunamadı.")

    repacked_path = path.rstrip('/\\') + '_repacked.pth'

    if not os.path.exists(repacked_path):
        print(f"[i] '{path}' klasörü .pth dosyasına dönüştürülüyor...")
        archive_name = 'archive'
        with zipfile.ZipFile(repacked_path, 'w', zipfile.ZIP_STORED) as zf:
            for root, _, files in os.walk(real_root):
                for fname in files:
                    full_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(full_path, real_root)
                    arcname = os.path.join(archive_name, rel_path).replace('\\', '/')
                    zf.write(full_path, arcname)

    return repacked_path


# --- MODEL YÜKLEME VE ÇALIŞTIRMA ---
model = BigramLanguageModel().to(device)
resolved_model_path = resolve_pth_path(BEST_MODEL_PATH)
state_dict = torch.load(resolved_model_path, map_location=device, weights_only=False)

state_dict = { (k.replace('_orig_mod.', '') if k.startswith('_orig_mod.') else k): v
               for k, v in state_dict.items() }
model.load_state_dict(state_dict)
model.eval()

print(f"\n[✓] Model yüklendi! ({BEST_MODEL_PATH})")
print("Başlangıç metnini gir (Prompt), model devamını tamamlasın.")
print("Çıkmak için 'exit' yazabilirsin.\n")

MAX_NEW_TOKENS = 40
TEMPERATURE = 0.10
TOP_K = 10
TOP_P = 0.70
REPETITION_PENALTY = 1.30

while True:
    try:
        prompt = input(">> Text: ")
    except (EOFError, KeyboardInterrupt):
        break

    stripped = prompt.strip()
    if stripped.lower() in ("exit", "quit", "q"):
        break

    if not stripped:
        continue

    print(f"\n>> {prompt}", end="")
    sys.stdout.flush()

    model.complete_text(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
    )