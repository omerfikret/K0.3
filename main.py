import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import glob
import json
import shutil
import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# ================== HYPERPARAMETRELER (TAM 42.2M PARAMETRE) ==================
# NOT: Mimari, LR, batch_size, dropout birebir aynı -> kalite/sonuç garanti
# olarak orijinaliyle aynı. DESIRED_EPOCHS artık aşağıda stage'e göre
# otomatik ayarlanıyor (bkz. "STAGE-FARKINDA EPOCH AYARI" bölümü).
batch_size     = 16
block_size     = 512
BASE_DESIRED_EPOCHS = 3   # stage 1'de kullanılan epoch sayısı; sonraki stage'lerde azalır
eval_interval  = None
learning_rate  = 6e-4
min_lr         = 6e-5
warmup_iters   = None
device         = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters     = 20

n_embd         = 512
n_head         = 8
n_layer        = 8
dropout        = 0.10
VOCAB_SIZE     = 16384
PATIENCE_EVALS = 15
max_iters      = None
# =========================================================================

torch.manual_seed(1337)

# --- T4 İÇİN GÜVENLİ/HIZLI BACKEND AYARLARI ---
# TF32 T4'te (Turing, cc7.5) desteklenmiyor, o yüzden açılmıyor (gereksiz).
# cudnn.benchmark zararsız: en hızlı algoritmayı otomatik seçer, sonucu değiştirmez.
torch.backends.cudnn.benchmark = True

# --- GOOGLE DRIVE'A KALICI KAYIT (KRİTİK: Colab local diski session kesilince sıfırlanır!) ---
# Tokenizer, tokenize edilmiş veri, shard'lar, checkpoint ve best_model artık
# Drive'da tutuluyor. Colab runtime tamamen kopsa/silinse bile bu dosyalar KALICI.
try:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = '/content/drive/MyDrive/llm_egitim'
except ImportError:
    # Colab dışında (yerel makine vb.) çalışıyorsa Drive'a gerek yok.
    BASE_DIR = '.'

os.makedirs(BASE_DIR, exist_ok=True)
print(f"[KALICI DEPO] Tüm checkpoint/veri şu klasöre yazılacak: {BASE_DIR}")

# --- DOSYA YOLLARI (prepare_data.py İLE SENKRON) ---
# DATA_FILE, prepare_data.py'deki FINAL_FILE ile BİREBİR AYNI yol/isim olmalı.
# PREPARE_STATE_FILE ise prepare_data.py'nin "kaçıncı stage'deyiz" hafızası;
# main.py bunu okuyup hem bilgilendirme yapıyor hem de epoch sayısını
# stage'e göre otomatik ayarlıyor (aşağıya bakın).
DATASETS_DIR = 'datasets'
DATA_FILE = os.path.join(DATASETS_DIR, 'huge_mixed_gutenberg.txt')
PREPARE_STATE_FILE = os.path.join(DATASETS_DIR, 'prepare_state.json')

TOKENIZER_FILE = os.path.join(BASE_DIR, 'tokenizer_16k.json')
DATA_CACHE = os.path.join(BASE_DIR, 'tokenized_huge_10m.pt')
SHARD_DIR = os.path.join(BASE_DIR, 'shards')
DATA_CACHE_META = os.path.join(BASE_DIR, 'data_cache_meta.json')  # cache hangi veriden üretildi, takip için
MODEL_DIR = os.path.join(BASE_DIR, 'model')
CHECKPOINT_PATH = os.path.join(MODEL_DIR, 'checkpoint.pth')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pth')

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"'{DATA_FILE}' bulunamadı. Önce prepare_data.py'yi çalıştırıp veri "
        f"hazırlamanız gerekiyor (örn. `python prepare_data.py --stage 1 "
        f"--target_tokens 200000000`)."
    )

# --- KAÇINCI STAGE'DEYİZ? (prepare_data.py'nin state dosyasından otomatik oku) ---
# Elle takip etmenize gerek kalmasın diye: prepare_data.py her stage'i
# tamamladığında bunu prepare_state.json'a yazıyor, biz burada sadece okuyoruz.
stage_count = 1
if os.path.exists(PREPARE_STATE_FILE):
    with open(PREPARE_STATE_FILE) as f:
        _prep_state = json.load(f)
    stage_count = max(1, len(_prep_state.get('completed_stages', [])))
    print(f"[STAGE] prepare_state.json'a göre şu ana kadar {stage_count} stage tamamlanmış.")
else:
    print("[STAGE] prepare_state.json bulunamadı, stage 1 varsayılıyor.")

# --- STAGE-FARKINDA EPOCH AYARI ---
# Kümülatif veri her stage'de büyüdüğü için (eski veri + yeni veri), sabit
# 3 epoch kullanmaya devam etmek eski veriyi orantısız fazla tekrar ettirir
# (stage 2'de eski veri 6. kez, stage 3'te 9. kez görülür gibi). Bunun yerine
# stage arttıkça epoch sayısını kademeli düşürüyoruz, taban 1 epoch'un altına
# inmiyor.
DESIRED_EPOCHS = max(1, BASE_DESIRED_EPOCHS - (stage_count - 1))
print(f"[EPOCH] Bu çalıştırmada kullanılacak epoch sayısı: {DESIRED_EPOCHS} "
      f"(stage {stage_count} -> taban {BASE_DESIRED_EPOCHS}'ten otomatik düşürüldü)")

# --- TOKENIZER YÜKLE / EĞİT ---
if os.path.exists(TOKENIZER_FILE):
    print("Kayıtlı tokenizer bulundu, yükleniyor...")
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
else:
    print(f"Tokenizer bulunamadı, '{DATA_FILE}' üzerinden eğitiliyor...")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=["<unk>"], show_progress=True)
    tokenizer.train([DATA_FILE], trainer)
    tokenizer.save(TOKENIZER_FILE)
    print(f"Tokenizer '{TOKENIZER_FILE}' olarak kaydedildi.")

vocab_size = tokenizer.get_vocab_size()
encode = lambda s: tokenizer.encode(s).ids
decode = lambda l: tokenizer.decode(l)

# --- VERİYİ TOKENIZE ET (RESUMABLE / SHARD BAZLI - Colab kesintisine dayanıklı) ---
CHUNK_LINES = 50_000

# --- ESKİ STAGE'DEN KALAN CACHE'İ OTOMATİK GEÇERSİZ KIL ---
# Önceki sürümde her yeni stage'den sonra tokenized_huge_10m.pt ve shards/
# klasörünü ELLE silmeniz gerekiyordu; unutulursa main.py sessizce eski
# (küçük) veriyle eğitime devam ediyordu -> yeni eklediğiniz veri hiç
# görülmüyordu. Artık DATA_FILE'ın boyutunu bir önceki çalıştırmayla
# karşılaştırıp, değiştiyse cache'i otomatik temizliyoruz.
current_data_size = os.path.getsize(DATA_FILE)
cache_is_stale = True
if os.path.exists(DATA_CACHE_META):
    with open(DATA_CACHE_META) as f:
        _meta = json.load(f)
    cache_is_stale = _meta.get('data_file_size') != current_data_size

if cache_is_stale and (os.path.exists(DATA_CACHE) or os.path.exists(SHARD_DIR)):
    print(f"[CACHE] '{DATA_FILE}' boyutu değişmiş (yeni stage verisi), "
          f"eski tokenize cache'i ve shard'lar temizleniyor...")
    if os.path.exists(DATA_CACHE):
        os.remove(DATA_CACHE)
    if os.path.exists(SHARD_DIR):
        shutil.rmtree(SHARD_DIR)

if os.path.exists(DATA_CACHE):
    print("Tokenize edilmiş veri cache'den yükleniyor...")
    data = torch.load(DATA_CACHE)
else:
    os.makedirs(SHARD_DIR, exist_ok=True)
    existing_shards = sorted(glob.glob(f'{SHARD_DIR}/shard_*.pt'))
    shard_idx = len(existing_shards)
    lines_already_done = shard_idx * CHUNK_LINES

    if shard_idx > 0:
        print(f"[RESUME] {shard_idx} shard zaten diskte bulundu, "
              f"~{lines_already_done:,} satır atlanıp kaldığı yerden devam edilecek.")

    print("Ham metin tokenize ediliyor (shard'lar halinde, kesinti-dayanıklı)...")
    buffer = []
    total_lines = 0
    last_report_m = lines_already_done // 1_000_000

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < lines_already_done:
                continue
            if line.strip():
                buffer.append(line)
            total_lines = i + 1

            if len(buffer) >= CHUNK_LINES:
                ids = []
                for enc in tokenizer.encode_batch(buffer):
                    ids.extend(enc.ids)
                shard_path = f'{SHARD_DIR}/shard_{shard_idx:05d}.pt'
                torch.save(torch.tensor(ids, dtype=torch.long), shard_path)
                shard_idx += 1
                buffer.clear()
                if total_lines // 1_000_000 > last_report_m:
                    last_report_m = total_lines // 1_000_000
                    print(f"  [+] {last_report_m}M satır tokenize edildi... "
                          f"({shard_idx} shard kaydedildi -> {shard_path})")

        if buffer:
            ids = []
            for enc in tokenizer.encode_batch(buffer):
                ids.extend(enc.ids)
            shard_path = f'{SHARD_DIR}/shard_{shard_idx:05d}.pt'
            torch.save(torch.tensor(ids, dtype=torch.long), shard_path)
            shard_idx += 1
            buffer.clear()

    print("Tüm shard'lar tokenize edildi, tek dosyada birleştiriliyor...")
    all_shards = sorted(glob.glob(f'{SHARD_DIR}/shard_*.pt'))
    data = torch.cat([torch.load(p) for p in all_shards])
    torch.save(data, DATA_CACHE)
    print(f"[✓] Tokenize işlemi bitti ({len(data):,} token) ve '{DATA_CACHE}' olarak kaydedildi.")
    # Cache oluştuktan sonra shard'lara artık ihtiyaç yok, disk yer açmak için silinebilir:
    # for p in all_shards: os.remove(p)

    with open(DATA_CACHE_META, 'w') as f:
        json.dump({'data_file_size': current_data_size, 'stage_count': stage_count}, f)

print(f"Toplam token sayısı: {len(data):,}  |  vocab_size: {vocab_size:,}")

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# --- VERİYİ GPU'DA TUT (T4 16GB'a rahat sığar, her batch'te CPU->GPU kopyalama maliyetini yok eder) ---
if device == 'cuda':
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    print(f"[HIZ] Eğitim/val verisi doğrudan GPU belleğinde tutuluyor "
          f"(~{(train_data.numel()+val_data.numel())*8/1e9:.2f} GB).")

# --- BÜYÜK VERİYE GÖRE DİNAMİK ADIM HESABI ---
tokens_per_iter = batch_size * block_size
steps_per_epoch = max(1, len(train_data) // tokens_per_iter)
max_iters = DESIRED_EPOCHS * steps_per_epoch
warmup_iters = max(100, int(0.03 * max_iters))
eval_interval = max(100, max_iters // 40)

print(f"Epoch başına adım: {steps_per_epoch:,}  |  Toplam max_iters: {max_iters:,}  ({DESIRED_EPOCHS} epoch)")
print(f"warmup_iters: {warmup_iters:,}  |  eval_interval: {eval_interval:,}")

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,), device=d.device)
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    if x.device.type != device:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x, y

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    torch.cuda.empty_cache()
    return out

# ================== MODEL MİMARİSİ ==================
# Matematiksel/mimari olarak orijinaliyle BİREBİR AYNI: n_head bağımsız Linear yerine
# tek bir büyük Linear(n_embd, 3*n_embd) kullanılıyor ve n_head parçaya bölünüyor.
# Bu, ayrı ayrı head'lerin concat edilmesiyle tamamen aynı kapasiteye/parametre
# sayısına sahip; sadece GPU'da tek büyük matmul olarak çalıştığı için (8 küçük
# matmul yerine) çok daha hızlı. Attention hesaplaması F.scaled_dot_product_attention
# ile yapılıyor (causal mask + softmax + dropout matematiksel olarak aynı, sadece
# T4'te de aktif olan memory-efficient kernel'i kullanıyor).
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
        qkv = self.qkv(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )  # (B, nh, T, hs)

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
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))

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
        x = self.blocks(tok_emb + pos_emb)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ================== EĞİTİM DÖNGÜSÜ ==================
os.makedirs(MODEL_DIR, exist_ok=True)

model = BigramLanguageModel()
m = model.to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parametre sayısı: {n_params/1e6:.2f}M")

use_fused = (device == 'cuda')
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, weight_decay=0.1, fused=use_fused
)

use_amp = (device == 'cuda')
scaler = torch.amp.GradScaler(enabled=use_amp)

best_val_loss = float('inf')
patience_counter = 0
start_iter = 0

# --- CHECKPOINT'TEN DEVAM ET (training kesilirse en baştan başlamamak için) ---
if os.path.exists(CHECKPOINT_PATH):
    print(f"[RESUME] '{CHECKPOINT_PATH}' bulundu, eğitime kaldığı yerden devam ediliyor...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scaler.load_state_dict(ckpt['scaler'])
    start_iter = ckpt['iter'] + 1
    best_val_loss = ckpt['best_val_loss']
    patience_counter = ckpt['patience_counter']
    print(f"[RESUME] iter={start_iter}, best_val_loss={best_val_loss:.4f} adımından devam.")

# torch.compile: aynı hesaplama grafiğini derleyip hızlandırır, sonucu değiştirmez.
try:
    model = torch.compile(model)
    print("[HIZ] Model torch.compile ile derlendi.")
except Exception as e:
    print(f"[UYARI] torch.compile başarısız oldu, derlemesiz devam ediliyor: {e}")

training_start = time.time()

for iter in range(start_iter, max_iters):
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter % 50 == 0:
        print(f"Adım: {iter}/{max_iters} tamamlanıyor... (LR: {lr:.6f})")

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"---> step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            patience_counter = 0
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(raw_model.state_dict(), BEST_MODEL_PATH)
            print(f"    [KAYIT] Yeni en iyi model Drive'a kaydedildi! (Best Val Loss: {best_val_loss:.4f})")
            print(f"    [KAYIT] Konum: {BEST_MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"    [SABIR] Val loss iyileşmedi. Sabır sayacı: {patience_counter}/{PATIENCE_EVALS}")

        # Her eval'de resume-checkpoint kaydet (Colab kesilirse kaldığı yerden devam için)
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        torch.save({
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'iter': iter,
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
        }, CHECKPOINT_PATH)
        print(f"    [KAYIT] Resume checkpoint Drive'a kaydedildi -> {CHECKPOINT_PATH}")

        if patience_counter >= PATIENCE_EVALS:
            print(f"\n[ERKEN DURDURMA] Val loss {PATIENCE_EVALS} değerlendirme boyunca iyileşmedi. "
                  f"Eğitim {iter}. adımda durduruluyor.")
            break

    xb, yb = get_batch('train')

    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
        logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    scaler.step(optimizer)
    scaler.update()

    if iter == start_iter + 100:
        elapsed = time.time() - training_start
        step_time = elapsed / 100
        eta_seconds = step_time * (max_iters - iter)
        print(f"\n[SÜRE TAHMİNİ] Kalan süre: ~{eta_seconds/3600:.2f} saat ({eta_seconds/60:.0f} dakika)\n")

total_time = time.time() - training_start
print(f"\nEğitim tamamlandı. Toplam süre: {total_time/3600:.2f} saat")

# asıl eğitim noktası