"""
prepare_data.py
================
Günlük/modern İngilizce dil verisi hazırlama - KADEMELİ (stage) sürekli eğitim için.
"""

import os, json, random, re, argparse
from datasets import load_dataset

OUT_DIR = "datasets"
STATE_FILE = os.path.join(OUT_DIR, "prepare_state.json")
FINAL_FILE = os.path.join(OUT_DIR, "huge_mixed_gutenberg.txt")
STAGE_DIR = os.path.join(OUT_DIR, "stages")

SOURCE_WEIGHTS = {
    "opensubtitles": 0.55,   # OpenOrca (Diyalog / Soru-Cevap / Sohbet)
    "bookcorpus":    0.03,   # emozilla/pg19 (Klasik Romanlar)
    "openwebtext":   0.22,   
    "cc_news":       0.20,   
}

WORD_TO_TOKEN_RATIO = 1.3  
MIN_DOC_CHARS = 20
PROGRESS_EVERY_WORDS = 2_000_000  


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
    """İlgili kaynak için streaming iterator döner."""
    if name == "openwebtext":
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        return (ex["text"] for ex in ds)
        
    if name == "bookcorpus":
        ds = load_dataset("emozilla/pg19", split="train", streaming=True)
        return (ex["text"] for ex in ds)
        
    if name == "cc_news":
        ds = load_dataset("vblagoje/cc_news", split="train", streaming=True)
        return (ex["text"] for ex in ds)
        
    if name == "opensubtitles":
        ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
        return (f"{ex['question']} {ex['response']}" for ex in ds)
        
    raise ValueError(name)


def collect_target_tokens(name, target_tokens, skip):
    """Hedef token sayısına ulaşana kadar yeni doküman toplar."""
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
        exhausted = True  

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

    # --- 1. AŞAMA: TALEP EDİLEN STAGE VERİSİNİ ÇEKME & KAYDETME ---
    if args.stage in state["completed_stages"]:
        print(f"[UYARI] Stage {args.stage} zaten tamamlanmış görünüyor. İndirme atlanıyor.")
    else:
        print(f"=== STAGE {args.stage}: hedef ~{args.target_tokens:,} token ===")
        stage_docs = []
        remaining_sources = dict(SOURCE_WEIGHTS)   
        pending_tokens = args.target_tokens        

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
                if got_words < target_words:   
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
        print(f"[OK] Stage {args.stage} indirildi ve kaydedildi -> {stage_path} ({len(stage_docs):,} doküman)")

        state["completed_stages"].append(args.stage)
        save_state(state)

    # --- 2. AŞAMA: TÜM STAGELERİ OKUMA, KARIŞTIRMA VE BİRLEŞTİRME ---
    print("\n=== Tüm Stage'ler Birleştiriliyor ve Karıştırılıyor ===")
    all_docs = []
    
    # Disk üzerindeki mevcut tüm stage_XX.txt dosyalarını bulup oku
    existing_stages = sorted(state["completed_stages"])
    for s in existing_stages:
        path = os.path.join(STAGE_DIR, f"stage_{s:02d}.txt")
        if os.path.exists(path):
            print(f"  -> {path} yükleniyor...")
            with open(path, encoding="utf-8") as f:
                all_docs.extend(line.rstrip("\n") for line in f if line.strip())

    # Bütün veriyi global olarak karıştır
    print(f"  -> Toplam {len(all_docs):,} doküman karma yapılıyor (random.shuffle)...")
    random.shuffle(all_docs)

    # Karıştırılmış veriyi nihai dosyaya yaz
    os.makedirs(os.path.dirname(FINAL_FILE), exist_ok=True)
    with open(FINAL_FILE, "w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(doc + "\n")

    total_words = sum(len(d.split()) for d in all_docs)
    print(f"\n[BAŞARILI] Tüm veriler karıştırıldı -> {FINAL_FILE}")
    print(f"           Toplam Doküman: {len(all_docs):,}")
    print(f"           Toplam Kelime : ~{total_words:,}")
    print(f"           Tahmini Token : ~{int(total_words*WORD_TO_TOKEN_RATIO):,}")


if __name__ == "__main__":
    main()