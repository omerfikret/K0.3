"""
prepare_data.py
================
Günlük/modern İngilizce dil verisi hazırlama - KADEMELİ (stage) sürekli eğitim için.

MANTIK:
  - Her stage'de her kaynaktan DAHA ÖNCE ÇEKİLMEMİŞ yeni dokümanlar indirilir
    (state dosyası hangi noktada kaldığını hatırlar, tekrar indirmez).
  - O ana kadarki TÜM stage'lerin verisi birleştirilip DOKÜMAN SEVİYESİNDE
    yeniden karıştırılır (shuffle) ve main.py'nin okuyacağı tek dosyaya yazılır.
  - Yani "önceki + yeni" mantığını script otomatik sağlıyor; main.py'ye her
    seferinde SADECE büyümüş kümülatif dosyayı gösteriyorsunuz.

KULLANIM (Kaggle notebook hücresinde):
    !pip install "datasets<4.0.0" --quiet
    !python prepare_data.py --stage 1 --target_tokens 200_000_000
    # main.py'yi çalıştırıp stage 1'i eğitin, sonra:
    !python prepare_data.py --stage 2 --target_tokens 200_000_000
    # main.py'yi TEKRAR çalıştırın (checkpoint'ten devam eder) ...

ÖNEMLİ - datasets kütüphanesi sürümü:
    Hugging Face `datasets` v4.0+ eski "script tabanlı" veri setlerini artık
    desteklemiyor (bookcorpusopen, cc_news, open_subtitles gibi). Bu yüzden
    `datasets<4.0.0` kurmanız gerekiyor. openwebtext (parquet formatında)
    yeni sürümde de çalışır ama tutarlılık için hepsini eski sürümle çekin.
"""

import os, json, random, re, argparse
from datasets import load_dataset

OUT_DIR = "datasets"
STATE_FILE = os.path.join(OUT_DIR, "prepare_state.json")
FINAL_FILE = os.path.join(OUT_DIR, "huge_mixed_gutenberg.txt")  # main.py'deki DATA_FILE ile AYNI isim/yol olmalı
STAGE_DIR = os.path.join(OUT_DIR, "stages")

# Günlük/konuşma diline ağırlık veren karışım oranları.
# opensubtitles indirilemezse (bkz. aşağıdaki try/except) payı otomatik
# openwebtext'e aktarılır, script durmaz.
SOURCE_WEIGHTS = {
    "opensubtitles": 0.30,   # gerçek diyalog -> en "günlük" kaynak
    "openwebtext":   0.35,   # blog/forum/makale -> doğal, güncel yazı dili
    "bookcorpus":    0.25,   # modern roman anlatısı + diyalog
    "cc_news":       0.10,   # haber dili -> resmi kayıt için az miktarda
}

WORD_TO_TOKEN_RATIO = 1.3  # BPE için kaba tahmin; gerçek sayıyı main.py tokenize ederken görürsünüz
MIN_DOC_CHARS = 20
PROGRESS_EVERY_WORDS = 2_000_000  # her 2M kelimede bir ilerleme yazdır


def clean_doc(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"opensubtitles": 0, "openwebtext": 0, "bookcorpus": 0, "cc_news": 0, "completed_stages": []}


def save_state(state):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def stream_docs(name):
    """İlgili kaynak için streaming iterator döner (tüm veri diske inmez, akış halinde okunur)."""
    if name == "openwebtext":
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        return (ex["text"] for ex in ds)
    if name == "bookcorpus":
        ds = load_dataset("bookcorpusopen", split="train", streaming=True, trust_remote_code=True)
        return (ex["text"] for ex in ds)
    if name == "cc_news":
        # Not: 'cc_news' (namespace'siz) artık yeni huggingface_hub ile çalışmıyor
        # ("Invalid HF URI... namespace/name" hatası). Namespace'li parquet
        # kopyasını kullanıyoruz - script gerekmiyor, daha da sağlam.
        ds = load_dataset("vblagoje/cc_news", split="train", streaming=True)
        return (ex["text"] for ex in ds)
    if name == "opensubtitles":
        ds = load_dataset("open_subtitles", lang1="en", lang2="tr", split="train",
                           streaming=True, trust_remote_code=True)
        return _group_subtitle_lines(ex["translation"]["en"] for ex in ds)
    raise ValueError(name)


def _group_subtitle_lines(line_iter, group_size=25):
    """OpenSubtitles tek tek satır (~5-10 kelime) veriyor; bunları MIN_DOC_CHARS
    filtresinden geçebilecek ve daha bağlamlı olacak şekilde N'li gruplar
    halinde birleştirip tek 'doküman' olarak veriyoruz."""
    buf = []
    for line in line_iter:
        line = line.strip()
        if line:
            buf.append(line)
        if len(buf) >= group_size:
            yield " ".join(buf)
            buf = []
    if buf:
        yield " ".join(buf)


def collect_target_tokens(name, target_tokens, skip):
    """`skip` kadar dokümanı atlar (önceki stage'lerde zaten kullanıldı),
    ardından hedef token sayısına ulaşana kadar yeni doküman toplar.
    Kaynak veri biterse (StopIteration), toplanan ne varsa onunla döner -
    eksik kalan miktarı çağıran taraf başka kaynaklara aktarır."""
    target_words = int(target_tokens / WORD_TO_TOKEN_RATIO)
    docs, collected_words, n_consumed = [], 0, skip
    last_report = 0
    exhausted = False
    it = stream_docs(name)
    for i, doc in enumerate(it):
        if i < skip:
            continue
        n_consumed += 1
        doc = clean_doc(doc)
        if len(doc) < MIN_DOC_CHARS:
            continue
        docs.append(doc)
        collected_words += len(doc.split())
        if collected_words - last_report >= PROGRESS_EVERY_WORDS:
            last_report = collected_words
            print(f"    ...[{name}] {collected_words:,}/{target_words:,} kelime")
        if collected_words >= target_words:
            break
    else:
        exhausted = True  # for-else: break olmadan biterse kaynak tükenmiş demektir

    if exhausted and collected_words < target_words:
        print(f"  [UYARI] '{name}' kaynağı tükendi, hedefin altında kaldı "
              f"({collected_words:,}/{target_words:,} kelime). Eksik pay diğer kaynaklara aktarılacak.")
    return docs, n_consumed, collected_words, target_words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--target_tokens", type=int, default=200_000_000)
    args = ap.parse_args()

    os.makedirs(STAGE_DIR, exist_ok=True)
    state = load_state()

    if args.stage in state["completed_stages"]:
        print(f"[UYARI] Stage {args.stage} zaten tamamlanmış, yeniden indirilmeyecek. "
              f"Mevcut tüm stage'ler birleştirilip {FINAL_FILE} yeniden yazılacak.")
    else:
        print(f"=== STAGE {args.stage}: hedef {args.target_tokens:,} token ===")
        stage_docs = []
        remaining_sources = dict(SOURCE_WEIGHTS)   # her turda tükenmeyenler kalır
        pending_tokens = args.target_tokens        # henüz karşılanmamış toplam hedef

        # En fazla len(SOURCE_WEIGHTS) tur atılır: her turda bir kaynak ya
        # hedefini tam karşılar ya da tükenip devre dışı kalır; kalan pay
        # bir sonraki turda hâlâ aktif olan kaynaklara yeniden dağıtılır.
        for _ in range(len(SOURCE_WEIGHTS)):
            if not remaining_sources or pending_tokens <= 0:
                break
            weight_sum = sum(remaining_sources.values())
            exhausted_this_round = []
            tokens_this_round = pending_tokens

            for source, weight in list(remaining_sources.items()):
                src_target = int(tokens_this_round * (weight / weight_sum))
                skip = state.get(source, 0)
                print(f"  [{source}] hedef ~{src_target:,} token | {skip:,} doküman atlanacak...")
                try:
                    docs, new_skip, got_words, target_words = collect_target_tokens(source, src_target, skip)
                except Exception as e:
                    print(f"  [HATA] '{source}' çekilemedi ({e}). Bu kaynak devre dışı bırakılıyor.")
                    exhausted_this_round.append(source)
                    continue
                print(f"  [{source}] {len(docs):,} yeni doküman toplandı "
                      f"(~{got_words:,}/{target_words:,} kelime).")
                stage_docs.extend(docs)
                state[source] = new_skip
                pending_tokens -= int(got_words * WORD_TO_TOKEN_RATIO)
                if got_words < target_words:   # kaynak tükendi, bir daha denemeye gerek yok
                    exhausted_this_round.append(source)

            for source in exhausted_this_round:
                remaining_sources.pop(source, None)

        if pending_tokens > 0:
            print(f"  [i] Tüm kaynaklar tüketildi, ~{pending_tokens:,} token hedefin altında kaldı.")

        random.shuffle(stage_docs)
        stage_path = os.path.join(STAGE_DIR, f"stage_{args.stage:02d}.txt")
        with open(stage_path, "w", encoding="utf-8") as f:
            for doc in stage_docs:
                f.write(doc + "\n")
        print(f"[OK] Stage {args.stage} kaydedildi -> {stage_path} ({len(stage_docs):,} doküman)")

        state["completed_stages"].append(args.stage)
        save_state(state)

    # --- Tamamlanmış TÜM stage'leri birleştir + yeniden karıştır (kümülatif veri) ---
    all_docs = []
    for s in sorted(state["completed_stages"]):
        path = os.path.join(STAGE_DIR, f"stage_{s:02d}.txt")
        with open(path, encoding="utf-8") as f:
            all_docs.extend(line.rstrip("\n") for line in f if line.strip())

    random.shuffle(all_docs)
    os.makedirs(os.path.dirname(FINAL_FILE), exist_ok=True)
    with open(FINAL_FILE, "w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(doc + "\n")

    total_words = sum(len(d.split()) for d in all_docs)
    print(f"\n[OK] Kümülatif veri hazır -> {FINAL_FILE}")
    print(f"     Doküman: {len(all_docs):,} | ~{total_words:,} kelime | "
          f"~{int(total_words*WORD_TO_TOKEN_RATIO):,} token (tahmini)")
    print("\n[SONRAKI ADIM] main.py'yi çalıştırmadan önce:")
    print("  1) tokenizer_16k.json'a DOKUNMAYIN (vocab sabit kalmalı, yoksa embedding'ler bozulur)")
    print("  2) tokenized_huge_10m.pt ve shards/ klasörünü SİLİN (yeni kümülatif veriyle yeniden tokenize etsin)")
    print("  3) checkpoint.pth ve best_model.pth'a DOKUNMAYIN (kaldığı ağırlıklardan devam etsin)")


if __name__ == "__main__":
    main()