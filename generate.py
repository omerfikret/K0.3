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
BEST_MODEL_PATH = 'K0.3.3/42M_model3_st800M_repacked.pth'

# --- TEST LOG DOSYASI ---
# Her "kaydet" dediğinde bu dosyanın SONUNA yeni bir başlıklı blok eklenir
# (Word dosyandaki "1. General Knowledge...", "2. Logic..." gibi bloklar
# tam olarak bu düzende birikir). Dosya yoksa otomatik oluşturulur.
TEST_LOG_FILE = 'test_sonuclari.txt'

# --- YAZI HIZI (typewriter efekti, saniye/karakter) ---
TYPE_DELAY = 0.008

if not os.path.exists(TOKENIZER_FILE):
    raise FileNotFoundError(f"Tokenizer bulunamadı: {TOKENIZER_FILE}")
if not os.path.exists(BEST_MODEL_PATH):
    raise FileNotFoundError(f"Eğitilmiş model bulunamadı: {BEST_MODEL_PATH}")

tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
vocab_size = tokenizer.get_vocab_size()
encode = lambda s: tokenizer.encode(s).ids
decode = lambda l: tokenizer.decode(l)

_SPECIAL_TOKEN_CANDIDATES = [
    "<unk>", "[UNK]", "<pad>", "[PAD]", "<s>", "</s>",
    "[BOS]", "[EOS]", "<|endoftext|>", "[CLS]", "[SEP]", "[MASK]",
]
_vocab = tokenizer.get_vocab()
BANNED_TOKEN_IDS = sorted({_vocab[t] for t in _SPECIAL_TOKEN_CANDIDATES if t in _vocab})

# ================== MODEL MİMARİSİ (main.py İLE BİREBİR AYNI) ==================
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
            q, k, v, is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out))

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
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
        Düz metin tamamlama (GPT-2 stili). SADECE üretir ve döner - ekrana
        basmaz, kayıt yapmaz. Yazdırma/kayıt işini dışarıdaki döngü yapıyor
        (böylece hem normal modda hem test modunda aynı typewriter/log
        mantığı tek yerden yönetiliyor).
        """
        prompt_ids = encode(prompt_text)
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated_ids = []

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if BANNED_TOKEN_IDS:
                logits[0, BANNED_TOKEN_IDS] = -float('Inf')

            if repetition_penalty is not None and repetition_penalty != 1.0:
                for prev_token in set(idx[0].tolist()):
                    if logits[0, prev_token] < 0:
                        logits[0, prev_token] *= repetition_penalty
                    else:
                        logits[0, prev_token] /= repetition_penalty

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

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
            generated_ids.append(idx_next.item())
            idx = torch.cat((idx, idx_next), dim=1)

        full_generated_text = decode(generated_ids)

        if stop_at_sentence:
            end_marks = ['.', '!', '?']
            last_mark_idx = max(full_generated_text.rfind(mark) for mark in end_marks)
            cleaned_text = full_generated_text[:last_mark_idx + 1] if last_mark_idx != -1 else full_generated_text
        else:
            cleaned_text = full_generated_text

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


def stream_print(text):
    """ChatGPT tarzı DEĞİL, GPT-2 demo tarzı: prompt'un TAM DEVAMI gibi akar.
    Ayrı bir '>>' satırı açmaz, kullanıcının yazdığı satırın hemen ardından
    (gerekirse aralarına boşluk koyarak) karakter karakter yazar."""
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(TYPE_DELAY)
    print("\n")


def save_test_block(title, entries, gen_params):
    """entries: [(prompt, output), ...]. Dosyanın SONUNA, Word'deki formatla
    birebir aynı düzende (başlık -> parametreler -> Inputs/Outputs) ekler."""
    with open(TEST_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{title}\n\n")
        f.write(f"MAX_NEW_TOKENS = {gen_params['max_new_tokens']}\n")
        f.write(f"TEMPERATURE = {gen_params['temperature']}\n")
        f.write(f"TOP_K = {gen_params['top_k']}\n")
        f.write(f"TOP_P = {gen_params['top_p']}\n")
        f.write(f"REPETITION_PENALTY = {gen_params['repetition_penalty']}\n\n")
        f.write("Outputs :\n\n")
        for prompt, output in entries:
            f.write(f">> Text: {prompt}\n\n")
            f.write(f">> {output}\n\n")
        f.write("=" * 70 + "\n\n")


# --- MODEL YÜKLEME ---
model = BigramLanguageModel().to(device)
resolved_model_path = resolve_pth_path(BEST_MODEL_PATH)
state_dict = torch.load(resolved_model_path, map_location=device, weights_only=False)
state_dict = {(k.replace('_orig_mod.', '') if k.startswith('_orig_mod.') else k): v
              for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

print(f"\n[✓] Model yüklendi! ({BEST_MODEL_PATH})")

# --- BURADAN AŞAĞISINI HER TEST TURUNDA SEN DEĞİŞTİRECEKSİN ---
MAX_NEW_TOKENS = 40
TEMPERATURE = 0.10
TOP_K = 10
TOP_P = 0.70
REPETITION_PENALTY = 1.30
# ----------------------------------------------------------------

gen_params = dict(max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
                   top_k=TOP_K, top_p=TOP_P, repetition_penalty=REPETITION_PENALTY)

test_mode = input("\nTest yapılacak mı? (e/h): ").strip().lower() in ("e", "evet", "y", "yes")

if test_mode:
    print(f"\n[TEST MODU] Sorular sorulacak, '{TEST_LOG_FILE}' dosyasına kaydedilecek.")
    print("Komutlar: 'kaydet' -> şu ana kadarki soruları başlıkla kaydet | 'exit' -> çıkış\n")
else:
    print("\nBaşlangıç metnini gir (Prompt), model devamını tamamlasın.")
    print("Çıkmak için 'exit' yazabilirsin.\n")

entries_buffer = []  # [(prompt, output), ...] - sadece test modunda dolduruluyor

while True:
    try:
        prompt = input(">> Text: ")
    except (EOFError, KeyboardInterrupt):
        break

    stripped = prompt.strip()
    low = stripped.lower()

    if low in ("exit", "quit", "q"):
        if test_mode and entries_buffer:
            confirm = input(f"[UYARI] {len(entries_buffer)} soru henüz kaydedilmedi. "
                             f"Kaydetmeden çıkılsın mı? (e/h): ").strip().lower()
            if confirm not in ("e", "evet", "y", "yes"):
                continue
        break

    if test_mode and low == "kaydet":
        if not entries_buffer:
            print("[i] Kaydedilecek soru yok, önce en az bir soru sor.\n")
            continue
        title = input("Başlık: ").strip()
        save_test_block(title, entries_buffer, gen_params)
        print(f"[✓] '{title}' başlığıyla {len(entries_buffer)} soru '{TEST_LOG_FILE}' dosyasına eklendi.\n")
        entries_buffer = []
        continue

    if not stripped:
        continue

    output_text = model.complete_text(
        stripped,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
    )

    # Prompt'la üretilen metin arasına, kelimeler birbirine yapışmasın diye
    # gerekiyorsa bir boşluk koy (GPT-2 demo'da da prompt+devam tek akış gibi görünür)
    needs_space = stripped and not stripped.endswith((' ', '\n')) and not output_text.startswith((' ', '\n', ',', '.', '!', '?'))
    stream_print((" " if needs_space else "") + output_text)

    if test_mode:
        entries_buffer.append((stripped, output_text))