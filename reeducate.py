"""
reeducate.py
============
main.py'ye HİÇ dokunulmadı, bu tamamen AYRI bir script.

NE İÇİN VAR:
main.py sıfırdan (ya da kendi yerel Drive checkpoint'inden) eğitim yapıyordu.
Bu script ise elle bir yerden getirdiğiniz DIŞARIDAN bir ağırlık dosyasını
(örn. "K0.3.1" klasörüne kaydettiğiniz önceki eğitimin .pth'i) başlangıç
noktası olarak alıp, YENİ (kümülatif) veriyle Kaggle üzerinde devam ediyor.
Kısacası: "reeducate" = var olan modeli, büyümüş veriyle yeniden eğit.

DENGELİ ÖĞRENME (catastrophic forgetting'i önleme) için 3 şey yapılıyor:
  1) Eğitim verisi main.py ile aynı mantıkla KÜMÜLATİF + KARIŞIK geliyor
     (prepare_data.py zaten eski+yeni veriyi karıştırıp tek dosyada veriyor)
     -> model sadece yeni veriyi değil, eskisini de görmeye devam ediyor.
  2) Learning rate baştan tam hızda (6e-4) başlamıyor. "Warm restart" ile
     orijinal tepe LR'nin bir kısmı kadar (RESTART_LR_SCALE) başlayıp kısa
     bir warmup + cosine decay ile iniyor -> zaten öğrenilmiş ağırlıkları
     büyük/agresif adımlarla bozmuyor.
  3) Epoch sayısı stage'e göre otomatik azalıyor (main.py'deki mantığın
     birebir aynısı) -> eski veri orantısız fazla tekrar edilmiyor,
     modelin çoğu bütçesi asıl YENİ veriye ayrılıyor.

KAGGLE'DA KULLANIM ADIMLARI:
  1) "K0.3.1" klasörünüzü (checkpoint.pth veya best_model.pth + tokenizer_16k.json
     içeren) bir Kaggle Dataset olarak yükleyin, notebook'a "Add Data" ile ekleyin.
  2) Aşağıdaki INITIAL_WEIGHTS_PATH ve TOKENIZER_SOURCE_PATH'i o dataset'in
     GERÇEK yoluna göre güncelleyin (Kaggle sol panelde dataset'e tıklayınca
     tam yolu görürsünüz, genelde '/kaggle/input/<dataset-adı>/...' şeklindedir).
  3) Yerelde prepare_data.py ile ürettiğiniz güncel 'datasets/' klasörünü
     (huge_mixed_gutenberg.txt + prepare_state.json) AYRI bir Kaggle Dataset
     olarak yükleyip ekleyin, DATA_FILE ve PREPARE_STATE_FILE yollarını ona
     göre güncelleyin.
  4) Notebook ayarlarından GPU (T4 x1 yeterli) açın, İnternet açık olsun
     gerekmiyor (veri zaten dataset olarak yüklü) ama kapalıysa da sorun yok.
  5) Çalıştırın. Bittiğinde /kaggle/working/model/ altındaki
     checkpoint_reeducate.pth ve best_model_reeducate.pth dosyalarını
     indirip yeni bir "K0.3.2" dataset'i olarak saklayın (main.py'deki
     akışla aynı disiplin: her aşamanın çıktısını indir, sakla).
"""

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

# ================== HYPERPARAMETRELER (main.py İLE BİREBİR AYNI MİMARİ) ==================
batch_size     = 16
block_size     = 512
BASE_DESIRED_EPOCHS = 3     # main.py'deki taban ile aynı mantık, stage'e göre otomatik düşer
learning_rate  = 6e-4       # orijinal (sıfırdan eğitimdeki) tepe LR - referans olarak duruyor
min_lr         = 6e-5
RESTART_LR_SCALE = 0.4      # <-- DENGELİ ÖĞRENMENİN ANAHTARI: warm-restart tepe LR'si
                             #     orijinalin YARISI kadar başlıyor, ağırlıkları agresif ezmiyor.
                             #     Model hâlâ kararsız/az öğrenmişse 0.6-0.7'ye çıkarabilirsiniz;
                             #     zaten iyi öğrenmişse 0.3-0.4'e düşürüp daha temkinli gidin.
device         = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters     = 20

n_embd         = 512
n_head         = 8
n_layer        = 8
dropout        = 0.10
VOCAB_SIZE     = 16384
PATIENCE_EVALS = 15
max_iters      = None
warmup_iters   = None
eval_interval  = None
# ==========================================================================================

torch.manual_seed(1337)
torch.backends.cudnn.benchmark = True

# --- KAGGLE İÇİN YOLLAR (Colab/Drive YOK, hepsi /kaggle/working ve /kaggle/input) ---
BASE_DIR = '/kaggle/working'
os.makedirs(BASE_DIR, exist_ok=True)

INITIAL_WEIGHTS_PATH  = '/kaggle/input/datasets/sarlnovax/800m-k0-3-3/checkpoint_reeducate_repacked.pth'
TOKENIZER_SOURCE_PATH = '/kaggle/input/datasets/sarlnovax/800m-k0-3-3/tokenizer_16k.json'
DATASETS_DIR          = '/kaggle/input/datasets/sarlnovax/800m-k0-3-3'

DATA_FILE = os.path.join(DATASETS_DIR, 'huge_mixed_gutenberg.txt')
PREPARE_STATE_FILE = os.path.join(DATASETS_DIR, 'prepare_state.json')

TOKENIZER_FILE = os.path.join(BASE_DIR, 'tokenizer_16k.json')
DATA_CACHE = os.path.join(BASE_DIR, 'tokenized_reeducate.pt')
SHARD_DIR = os.path.join(BASE_DIR, 'shards_reeducate')
DATA_CACHE_META = os.path.join(BASE_DIR, 'data_cache_meta_reeducate.json')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
CHECKPOINT_PATH = os.path.join(MODEL_DIR, 'checkpoint_reeducate.pth')   # main.py'nin checkpoint.pth'ından
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model_reeducate.pth')  # kasıtlı FARKLI isim - karışmasın
os.makedirs(MODEL_DIR, exist_ok=True)

for p, desc in [(INITIAL_WEIGHTS_PATH, "başlangıç ağırlığı (K0.3.1)"),
                (TOKENIZER_SOURCE_PATH, "tokenizer"),
                (DATA_FILE, "eğitim verisi")]:
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"'{p}' bulunamadı ({desc}). Kaggle'da 'Add Data' ile doğru dataset'i "
            f"eklediğinizden ve dosya yollarını script'in başında güncellediğinizden emin olun."
        )

# Tokenizer'ı çalışma dizinine kopyala (vocab SABİT kalmalı, K0.3.1'de eğitileni kullanıyoruz)
if not os.path.exists(TOKENIZER_FILE):
    shutil.copy(TOKENIZER_SOURCE_PATH, TOKENIZER_FILE)
    print(f"[i] Tokenizer kopyalandı: {TOKENIZER_SOURCE_PATH} -> {TOKENIZER_FILE}")

# --- KAÇINCI STAGE'DEYİZ? (main.py'deki AYNI mantık, senkron kalsın diye) ---
stage_count = 1
if os.path.exists(PREPARE_STATE_FILE):
    with open(PREPARE_STATE_FILE) as f:
        _prep_state = json.load(f)
    stage_count = max(1, len(_prep_state.get('completed_stages', [])))
print(f"[STAGE] Şu ana kadar {stage_count} stage tamamlanmış -> bu 'reeducate' turu stage {stage_count} verisiyle çalışıyor.")

DESIRED_EPOCHS = max(1, BASE_DESIRED_EPOCHS - (stage_count - 1))
print(f"[EPOCH] Bu reeducate turunda kullanılacak epoch sayısı: {DESIRED_EPOCHS}")

# --- TOKENIZER YÜKLE (asla yeniden eğitilmiyor, K0.3.1'deki sabit vocab kullanılıyor) ---
tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
vocab_size = tokenizer.get_vocab_size()
if vocab_size != VOCAB_SIZE:
    print(f"[UYARI] Tokenizer vocab_size={vocab_size}, script'teki VOCAB_SIZE={VOCAB_SIZE} ile "
          f"uyuşmuyor. VOCAB_SIZE sadece bilgi amaçlı, gerçek boyut olarak tokenizer'dakini kullanıyoruz.")

# --- VERİYİ TOKENIZE ET (RESUMABLE / SHARD BAZLI - main.py ile aynı yaklaşım) ---
CHUNK_LINES = 50_000
current_data_size = os.path.getsize(DATA_FILE)
cache_is_stale = True
if os.path.exists(DATA_CACHE_META):
    with open(DATA_CACHE_META) as f:
        _meta = json.load(f)
    cache_is_stale = _meta.get('data_file_size') != current_data_size

if cache_is_stale and (os.path.exists(DATA_CACHE) or os.path.exists(SHARD_DIR)):
    print("[CACHE] Veri değişmiş, eski tokenize cache'i temizleniyor...")
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
        print(f"[RESUME] {shard_idx} shard zaten diskte, ~{lines_already_done:,} satır atlanacak.")

    print("Ham metin tokenize ediliyor (shard'lar halinde)...")
    buffer, total_lines = [], 0
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
                torch.save(torch.tensor(ids, dtype=torch.long), f'{SHARD_DIR}/shard_{shard_idx:05d}.pt')
                shard_idx += 1
                buffer.clear()
        if buffer:
            ids = []
            for enc in tokenizer.encode_batch(buffer):
                ids.extend(enc.ids)
            torch.save(torch.tensor(ids, dtype=torch.long), f'{SHARD_DIR}/shard_{shard_idx:05d}.pt')

    print("Tüm shard'lar birleştiriliyor...")
    all_shards = sorted(glob.glob(f'{SHARD_DIR}/shard_*.pt'))
    data = torch.cat([torch.load(p) for p in all_shards])
    torch.save(data, DATA_CACHE)
    with open(DATA_CACHE_META, 'w') as f:
        json.dump({'data_file_size': current_data_size, 'stage_count': stage_count}, f)

print(f"Toplam token sayısı: {len(data):,}  |  vocab_size: {vocab_size:,}")

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
if device == 'cuda':
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    print(f"[HIZ] Veri GPU belleğinde (~{(train_data.numel()+val_data.numel())*8/1e9:.2f} GB).")

# --- BÜYÜK VERİYE GÖRE DİNAMİK ADIM HESABI (bu reeducate turu için YENİ bir bütçe) ---
tokens_per_iter = batch_size * block_size
steps_per_epoch = max(1, len(train_data) // tokens_per_iter)
max_iters = DESIRED_EPOCHS * steps_per_epoch
warmup_iters = max(100, int(0.03 * max_iters))
eval_interval = max(100, max_iters // 40)

RESTART_PEAK_LR = learning_rate * RESTART_LR_SCALE
RESTART_MIN_LR = min_lr * RESTART_LR_SCALE

print(f"Epoch başına adım: {steps_per_epoch:,}  |  Toplam max_iters: {max_iters:,} ({DESIRED_EPOCHS} epoch)")
print(f"warmup_iters: {warmup_iters:,}  |  eval_interval: {eval_interval:,}")
print(f"[WARM-RESTART] Tepe LR: {RESTART_PEAK_LR:.6f} (orijinalin %{int(RESTART_LR_SCALE*100)}'i)  |  Min LR: {RESTART_MIN_LR:.6f}")


def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,), device=d.device)
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    if x.device.type != device:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x, y


def get_lr(it):
    """main.py'deki cosine schedule ile AYNI şekil, sadece tepe/min noktaları
    RESTART_LR_SCALE ile düşürülmüş -> 'warm restart', ağırlıkları ezmeden devam."""
    if it < warmup_iters:
        return RESTART_PEAK_LR * it / warmup_iters
    if it > max_iters:
        return RESTART_MIN_LR
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return RESTART_MIN_LR + coeff * (RESTART_PEAK_LR - RESTART_MIN_LR)


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

# ================== EĞİTİM DÖNGÜSÜ ==================
model = BigramLanguageModel()
m = model.to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parametre sayısı: {n_params/1e6:.2f}M")

use_fused = (device == 'cuda')
optimizer = torch.optim.AdamW(model.parameters(), lr=RESTART_PEAK_LR, weight_decay=0.1, fused=use_fused)
use_amp = (device == 'cuda')
scaler = torch.amp.GradScaler(enabled=use_amp)

best_val_loss = float('inf')
patience_counter = 0
start_iter = 0

# --- AĞIRLIKLARI YÜKLE: önce BU turun kendi (kesinti sonrası) checkpoint'i var mı bak,
#     yoksa K0.3.1'deki dış başlangıç ağırlığından başla. ---
if os.path.exists(CHECKPOINT_PATH):
    print(f"[RESUME] Bu reeducate turu zaten başlamış, '{CHECKPOINT_PATH}'den devam ediliyor...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scaler.load_state_dict(ckpt['scaler'])
    start_iter = ckpt['iter'] + 1
    best_val_loss = ckpt['best_val_loss']
    patience_counter = ckpt['patience_counter']
    print(f"[RESUME] iter={start_iter}, best_val_loss={best_val_loss:.4f}")
else:
    print(f"[YÜKLE] K0.3.1'den başlangıç ağırlığı yükleniyor: {INITIAL_WEIGHTS_PATH}")
    raw = torch.load(INITIAL_WEIGHTS_PATH, map_location=device, weights_only=False)
    if isinstance(raw, dict) and 'model' in raw:
        # Tam checkpoint.pth (model+optimizer+scaler) verilmiş -> hepsini devral,
        # bu en dengeli devam şekli (optimizer'ın momentum'u da korunur).
        state_dict = raw['model']
        print("[i] Tam checkpoint formatı algılandı (model+optimizer+scaler).")
    else:
        # Sadece ağırlık (best_model.pth tarzı, çıplak state_dict) verilmiş ->
        # optimizer/scaler sıfırdan başlıyor, bu da zaten warm-restart LR'siyle
        # tutarlı (agresif olmayan bir başlangıç).
        state_dict = raw
        print("[i] Çıplak state_dict formatı algılandı (sadece model ağırlıkları).")

    state_dict = {(k.replace('_orig_mod.', '') if k.startswith('_orig_mod.') else k): v
                  for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print("[✓] Başlangıç ağırlıkları modele yüklendi. Eğitim iter=0'dan (bu turun kendi bütçesiyle) başlıyor.")

try:
    model = torch.compile(model)
    print("[HIZ] Model torch.compile ile derlendi.")
except Exception as e:
    print(f"[UYARI] torch.compile başarısız: {e}")

training_start = time.time()

for iter in range(start_iter, max_iters):
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter % 50 == 0:
        print(f"Adım: {iter}/{max_iters} (LR: {lr:.6f})")

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"---> step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            patience_counter = 0
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(raw_model.state_dict(), BEST_MODEL_PATH)
            print(f"    [KAYIT] Yeni en iyi model -> {BEST_MODEL_PATH} (Best Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"    [SABIR] Val loss iyileşmedi. Sabır: {patience_counter}/{PATIENCE_EVALS}")

        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        torch.save({
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'iter': iter,
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
        }, CHECKPOINT_PATH)
        print(f"    [KAYIT] Resume checkpoint -> {CHECKPOINT_PATH}")

        if patience_counter >= PATIENCE_EVALS:
            print(f"\n[ERKEN DURDURMA] Val loss {PATIENCE_EVALS} değerlendirme boyunca iyileşmedi. Durduruluyor.")
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

    if iter == start_iter + eval_iters:
        elapsed = time.time() - training_start
        eta_seconds = (elapsed / 100) * (max_iters - iter)
        print(f"\n[SÜRE TAHMİNİ] Kalan süre: ~{eta_seconds/3600:.2f} saat\n")

total_time = time.time() - training_start
print(f"\nReeducate eğitimi tamamlandı. Toplam süre: {total_time/3600:.2f} saat")
print(f"\n[ÖNEMLİ] Kaggle oturumu kapanınca /kaggle/working silinir!")
print(f"  İndirmeyi unutmayın: {CHECKPOINT_PATH}  ve  {BEST_MODEL_PATH}")
print(f"  Bir sonraki stage için bunları yeni bir Kaggle Dataset ('K0.3.2' gibi) olarak saklayın.")