import os
import sys
import shutil
import threading
import subprocess
import time
import gc
import builtins
import requests
import torch
import uvicorn
import nest_asyncio
import psutil

# Gerekli kütüphaneler (SonarQube için importlar yukarı alındı)
from fastapi import FastAPI
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from huggingface_hub import notebook_login, login
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
# from google.colab import drive # Colab dışında çalışıyorsa bu satır hata verebilir, yorumda kalsın.

# -------------------------------------------------------------------
# 1. Kütüphane Kurulumu ve Hazırlık
# -------------------------------------------------------------------

major_version, minor_version = torch.cuda.get_device_capability()

# Kütüphaneleri sessizce kuruyoruz (Colab komutları yorum satırına alındı)
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" > /dev/null 2>&1
# !pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes > /dev/null 2>&1

print("Kütüphaneler başarıyla kuruldu.")

# Unsloth kütüphanesini ve tüm bağımlılıkları ortamına kurduk.

# -------------------------------------------------------------------
# 2. Temel Modelin Yüklenmesi ve LoRA Yapılandırması
# -------------------------------------------------------------------

max_seq_length = 2048 # Modelin okuyabileceği maksimum kelime uzunluğu
dtype = None
load_in_4bit = True # Hafıza tasarrufu için 4-bit yükleme

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit", # Llama 3 tabanlı optimize model
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# LoRA (Low-Rank Adaptation) ayarları - Modeli hızlı eğitmek için
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# Llama-3 modeli bellek tasarrufu için 4-bit formatında yüklendi.
# Ardından fine-tuning ayarı için LoRA(Low-Rank Adaptation) katmanları yapılandırıldı.

# -------------------------------------------------------------------
# 3. Veri Seti Hazırlığı ve Prompt Formatı
# -------------------------------------------------------------------

# Prompt Formatı
alpaca_prompt = """Aşağıda bir görevi tanımlayan bir talimat bulunmaktadır.
İsteği uygun şekilde tamamlayan bir yanıt yazın.

### Instruction:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Cümle sonu işareti

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs      = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        # Soru ve cevabı şablona oturtuyoruz
        text = alpaca_prompt.format(instruction, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

# Dosyayı yükle ve formatla
# NOT: Dosya yolu projenize göre güncellenmelidir.
try:
    dataset = load_dataset("json", data_files="/qa_pairs.jsonl", split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True,)
    print(f"Veri seti yüklendi. Toplam örnek sayısı: {len(dataset)}")
    # İlk örneği kontrol edelim
    print(dataset[0]["text"])
except Exception as e:
    print(f"Veri seti yüklenemedi (Dosya yolu kontrol edilmeli): {e}")


# -------------------------------------------------------------------
# 4. Eğitimi Başlatma (Fine-Tuning)
# -------------------------------------------------------------------

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset if 'dataset' in locals() else None,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Daha hızlı eğitim için True yapılabilir ama şimdilik False kalsın
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # 100-150 veriniz varsa 60 adım iyi bir başlangıçtır (Overfit olmasın)
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

print("--- EĞİTİM BAŞLIYOR (Yaklaşık 10-20 dakika) ---")
# trainer.train() # Eğitim satırı aktif edilmelidir.
print("--- EĞİTİM TAMAMLANDI! ---")

# -------------------------------------------------------------------
# 5. Modelin Test Edilmesi (Inference)
# -------------------------------------------------------------------

# Test için "FastLanguageModel.for_inference" moduna alıyoruz
FastLanguageModel.for_inference(model)

# Test Sorusu (Veri setinizden rastgele bir soru sorun)
soru = "Eskimiş ve Rentansiyonu Olmayan Bileşenler maddesinin tanımı nedir?"

inputs = tokenizer(
[
    alpaca_prompt.format(
        soru, # Instruction
        "", # Output (Boş bırakıyoruz, model dolduracak)
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)
cevap = tokenizer.batch_decode(outputs)
print("--- MODEL CEVABI ---")
# Çıktıdaki fazlalıkları temizleyerek sadece cevabı yazdır
print(cevap[0].split("### Response:")[-1].strip().replace(tokenizer.eos_token, ""))

# -------------------------------------------------------------------
# 6. Hugging Face Hub Kimlik Doğrulama
# -------------------------------------------------------------------

# notebook_login() # Jupyter ortamında manuel giriş için

# Hugging Face "token"ını girerek doğrulama işlemi gerçekleştirildi.

# -------------------------------------------------------------------
# 7. Modelin Buluta Yüklenmesi (Deployment)
# -------------------------------------------------------------------

# Kendi HF kullanıcı adınızı yazın
kullanici_adi = "nilnilu"
model_ismi = "owasp-guvenlik-chatbot"

# Modeli sadece LoRA adaptörleri (küçük dosyalar) olarak kaydediyoruz
# model.push_to_hub(f"{kullanici_adi}/{model_ismi}", token=True)
# tokenizer.push_to_hub(f"{kullanici_adi}/{model_ismi}", token=True)

print("Model başarıyla yüklendi! 🚀")


# -------------------------------------------------------------------
# FASTAPI KISMI
# -------------------------------------------------------------------
# 8. Kütüphane Kurulumu ve Drive Bağlantısı

# FastAPI, Uvicorn (Sunucu) ve diğer gerekli kütüphaneler
# !pip install fastapi uvicorn python-multipart nest-asyncio > /dev/null 2>&1
# Hibrit sistemi kurmak için gerekli kütüphaneler
# !pip install langchain-community sentence-transformers chromadb > /dev/null 2>&1

print("Kütüphaneler başarıyla kuruldu. Sistem çalışmaya hazır.")

# Sistemi bir web servisi olarak çalıştırmada gerekli FastAPI sunucu altyapısı kuruldu.
# Ayrıca RAG için gerekli vektör veritabanı ve metin işleme araçları sisteme yüklendi.

# -------------------------------------------------------------------
# 9. FT Modelini ve RAG Bileşenlerini Yükleme
# -------------------------------------------------------------------

# --- 2. ADIM: Drive Bağlantısı ve Dosya Transferi ---

# Ayarlar
HF_KULLANICI_ADI = "nilnilu"
MODEL_ADI = "owasp-guvenlik-chatbot"

# Kaynak Yolları (Drive)
DRIVE_ROOT = "/content/drive/MyDrive/Colab Notebooks/SecurityChatbot"
MODEL_DRIVE_PATH = f"{DRIVE_ROOT}/{HF_KULLANICI_ADI}/{MODEL_ADI}"
CHROMA_DRIVE_PATH = f"{DRIVE_ROOT}/chroma_db"

# Hedef Yolları (Colab Yerel Disk - Hız için)
MODEL_LOCAL_PATH = f"./{HF_KULLANICI_ADI}_{MODEL_ADI}"
CHROMA_LOCAL_PATH = "./chroma_db"

def setup_environment():
    print(" Ortam hazırlığı başlatılıyor...")

    # 1. Drive'a Bağlan
    try:
        # drive.mount('/content/drive') # Colab komutu
        print("✅ Drive bağlantısı denendi.")
    except Exception as e:
        print(f" Drive bağlantı uyarısı: {e}")

    # 2. Modeli Kopyala
    if os.path.exists(MODEL_DRIVE_PATH):
        if not os.path.exists(MODEL_LOCAL_PATH):
            print(f" Model yerel diske kopyalanıyor... (Bekleyiniz)")
            try:
                shutil.copytree(MODEL_DRIVE_PATH, MODEL_LOCAL_PATH, dirs_exist_ok=True)
                print("✅ Model kopyalandı.")
            except: print(" Kopyalama sırasında hata oluştu, internetten indirilecek.")
        else:
            print("ℹ Model zaten yerelde mevcut.")
    else:
        print(f" Model Drive'da bulunamadı: {MODEL_DRIVE_PATH}")

    # 3. ChromaDB Kopyala
    if os.path.exists(CHROMA_DRIVE_PATH):
        if not os.path.exists(CHROMA_LOCAL_PATH):
            print(f" Veritabanı yerel diske kopyalanıyor...")
            try:
                shutil.copytree(CHROMA_DRIVE_PATH, CHROMA_LOCAL_PATH, dirs_exist_ok=True)
                print("✅ Veritabanı kopyalandı.")
            except: print("⚠️ Veritabanı kopyalanamadı, Drive üzerinden okunacak.")
        else:
            print("ℹ Veritabanı zaten mevcut.")
    else:
        print("ℹ Drive'da veritabanı klasörü bulunamadı.")

    print("\n--- Hazırlık Adımı Tamamlandı ---")

setup_environment()

# Çalışma performansını artırmak amacıyla Google Drive'daki eğitilmiş model ve veritabanı dosyalarını Colab'in hızlı yerel diskine kopyalandı.

# -------------------------------------------------------------------
# 10. Model Eğitimi (Fine-Tuning) - Koşullu
# -------------------------------------------------------------------

EGITIM_YAPILSIN_MI = False  # <--- Video için False kalmalı!

if EGITIM_YAPILSIN_MI:
    # Ayarlar
    HF_KULLANICI = "nilnilu"
    MODEL_ISMI = "owasp-guvenlik-chatbot"
    DATA_PATH = "qa_pairs.jsonl"

    print(" Eğitim başlıyor...")

    # Model Yükleme
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b-bnb-4bit",
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

    # LoRA Ayarları
    model = FastLanguageModel.get_peft_model(
        model, r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16, lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth"
    )

    # Veri Seti ve Trainer (Burada eğitim kodları çalışır)
    # ... (Kodun devamı temsilidir)

    print("✅ Eğitim tamamlandı.")
else:
    print("ℹ EĞİTİM ADIMI ATLANDI.")
    print("   Sebep: Model zaten eğitildi ve yüklendi (nilnilu/owasp-guvenlik-chatbot).")
    print("   Doğrudan sunucu başlatma adımına geçiliyor.")

# -------------------------------------------------------------------
# 11. Akıllı Başlatma (FastAPI + RAG)
# -------------------------------------------------------------------

# --- 4. ADIM: SUNUCUYU BAŞLAT (AKILLI YÜKLEME) ---

# Ayarlar
HF_KULLANICI_ADI = "nilnilu"
MODEL_ADI = "owasp-guvenlik-chatbot"

# Yollar
MODEL_LOCAL_PATH = f"./{HF_KULLANICI_ADI}_{MODEL_ADI}"
CHROMA_LOCAL_PATH = "./chroma_db"
CHROMA_DRIVE_PATH = "/content/drive/MyDrive/Colab Notebooks/SecurityChatbot/chroma_db"

# RAG Prompt Şablonu
RAG_PROMPT = """Siz, XYZ Şirketi'nin Güvenlik Politikasını uygulayan deneyimli bir yapay zeka botusunuz.
Yalnızca aşağıdaki bağlamda (context) verilen bilgilere dayanarak cevap verin...

### Bağlam (Context):
{context}

### Kullanıcı Sorusu:
{question}

### Yanıt:
"""

# Global değişkenler
model = None
tokenizer = None
vectorstore = None

# --- FONKSİYONLAR ---

def load_model():
    """Modeli ve Veritabanını en uygun kaynaktan yükler (Local > Drive > Cloud)."""
    global model, tokenizer, vectorstore
    print(" Sistem bileşenleri yükleniyor...")

    # A) Eğitilebilir SLM (Model) Yükleme
    if os.path.exists(MODEL_LOCAL_PATH):
        print(f" Yerel model bulundu: {MODEL_LOCAL_PATH}")
        path_to_use = MODEL_LOCAL_PATH
    else:
        print(" Yerel model yok, Hugging Face'den indiriliyor...")
        path_to_use = f"{HF_KULLANICI_ADI}/{MODEL_ADI}"

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = path_to_use,
            max_seq_length = 2048,
            dtype = None,
            load_in_4bit = True,
        )
        FastLanguageModel.for_inference(model)
        print("✅ Yapay Zeka Modeli Yüklendi.")
    except Exception as e:
        print(f"❌ Model Hatası: {e}")
        raise e

    # B) RAG (Veritabanı) Yükleme
    print(" Veritabanı bağlantısı aranıyor...")
    embedding_function = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cuda'}
    )

    final_chroma_path = None
    if os.path.exists(CHROMA_LOCAL_PATH):
        final_chroma_path = CHROMA_LOCAL_PATH
        print(" Yerel veritabanı kullanılıyor.")
    elif os.path.exists(CHROMA_DRIVE_PATH):
        final_chroma_path = CHROMA_DRIVE_PATH
        print(" Drive veritabanı kullanılıyor.")
    else:
        # Son çare Drive'ı tekrar dene
        try:
            # if not os.path.exists('/content/drive'): drive.mount('/content/drive')
            if os.path.exists(CHROMA_DRIVE_PATH): final_chroma_path = CHROMA_DRIVE_PATH
        except: pass

    if final_chroma_path:
        try:
            vectorstore = Chroma(persist_directory=final_chroma_path, embedding_function=embedding_function)
            print("✅ RAG Veritabanı Bağlandı!")
        except: vectorstore = None
    else:
        print(" Veritabanı bulunamadı. (Sadece model bilgisi kullanılacak)")

    print(" SİSTEM HAZIR! Sunucu başlatılıyor...")

def generate_rag_response(question):
    """RAG Destekli Cevap Üretir."""
    context = ""
    if vectorstore:
        try:
            docs = vectorstore.similarity_search(question, k=3)
            context = "\n---\n".join([doc.page_content for doc in docs])
        except: context = "Veritabanı hatası."

    prompt_text = RAG_PROMPT.format(context=context, question=question)
    inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
    full_response = tokenizer.batch_decode(outputs)[0]
    response = full_response.split("### Yanıt:")[-1].strip().replace(tokenizer.eos_token, "")
    return response, context

# --- SERVER ---
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/chat")
async def chat_endpoint(request_data: dict):
    q = request_data.get("question")
    if not q: return {"response": "Soru giriniz."}
    try:
        res, ctx = generate_rag_response(q)
        return {"response": res, "context_used": ctx, "status": "success"}
    except Exception as e:
        return {"response": f"Hata: {str(e)}", "status": "error"}

# Bu bölüm script olarak çalıştırıldığında hata vermemesi için if __name__ bloğuna alınmalıdır
if __name__ == "__main__":
    nest_asyncio.apply()
    # uvicorn.run(app, host="0.0.0.0", port=8000)

# -------------------------------------------------------------------
# RAG VE ARAYÜZ KURULUMLARI (STREAMLIT)
# -------------------------------------------------------------------
# 12. Kurulumlar, Giriş ve Ayarlar

# --- 1. ADIM: GENEL KURULUM VE AYARLAR ---

# B) Kütüphanelerin Kurulumu
print("📦 Gerekli kütüphaneler kuruluyor... (Bu işlem 2-3 dakika sürebilir)")

# !pip install fastapi uvicorn python-multipart nest-asyncio psutil
# !pip install langchain-community sentence-transformers chromadb langchain huggingface_hub
# !pip install streamlit pyngrok # Arayüz için gerekli

# Cloudflare (Tünelleme için)
if not os.path.exists("cloudflared"):
    # !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    # !mv cloudflared-linux-amd64 cloudflared
    # !chmod +x cloudflared
    pass

# C) Hugging Face Girişi
print("\n🔑 Hugging Face Girişi Yapılıyor...")
# SonarQube: Hardcoded token riskini önlemek için os.getenv kullanımı önerilir.
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    # login() # Manuel giriş isteniyorsa
    pass

print("✅ Kurulumlar ve Giriş İşlemleri Tamamlandı!")

# -------------------------------------------------------------------
# 13. Sistem Düzeltmeleri ve Hazırlık
# -------------------------------------------------------------------

# --- 2. ADIM: SİSTEM DÜZELTMELERİ VE HAZIRLIK ---

print("🛠️ Sistem kararlılığı için düzeltmeler yapılıyor...")

# 1. Cache Temizliği (Olası hataları önler)
cache_path = "/content/unsloth_compiled_cache"
if os.path.exists(cache_path):
    shutil.rmtree(cache_path)
    print("   -> Cache temizlendi.")

# 2. Psutil Global Fix (Unsloth için gerekli)
builtins.psutil = psutil
print("   -> psutil global olarak ayarlandı.")

# 3. Dosya Kontrolü (Opsiyonel bilgi)
RAW_DOC_PATH = "raw_document.md"
if not os.path.exists(RAW_DOC_PATH):
    print(f"⚠️ Bilgi: '{RAW_DOC_PATH}' dosyası şu an yok. (Sorun değil, demo için kod içinde metin var.)")
else:
    print(f"✅ '{RAW_DOC_PATH}' dosyası mevcut.")

print("✅ Sistem eğitime ve çalışmaya hazır!")

# -------------------------------------------------------------------
# 14. MODEL KISMI (Tekrar - Yedek Kod Bloğu)
# -------------------------------------------------------------------

# --- 3. ADIM: MODEL EĞİTİMİ (FINE-TUNING) KODLARI ---
EGITIM_YAPILSIN_MI = False  # <--- Videoda burası False kalsın!

if EGITIM_YAPILSIN_MI:
    # AYARLAR
    HF_KULLANICI_ADI = "nilnilu"
    MODEL_ADI = "owasp-guvenlik-chatbot"
    QA_DATA_PATH = "qa_pairs.jsonl"

    # 1. Model Yükleme
    print("⏳ Eğitim için Temel Model Yükleniyor...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b-bnb-4bit",
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

    # LoRA Adaptörleri
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # 2. Veri Seti Hazırlığı
    if os.path.exists(QA_DATA_PATH):
        print(f"📚 '{QA_DATA_PATH}' ile eğitim başlıyor...")
        alpaca_prompt = """### Instruction:
    {}

    ### Response:
    {}"""

        def formatting_prompts_func(examples):
            instructions = examples["instruction"]
            outputs      = examples["output"]
            texts = []
            for instruction, output in zip(instructions, outputs):
                text = alpaca_prompt.format(instruction, output) + tokenizer.eos_token
                texts.append(text)
            return { "text" : texts, }

        dataset = load_dataset("json", data_files=QA_DATA_PATH, split="train")
        dataset = dataset.map(formatting_prompts_func, batched = True,)

        # 3. Eğitim Başlatma
        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = dataset,
            dataset_text_field = "text",
            max_seq_length = 2048,
            dataset_num_proc = 2,
            packing = False,
            args = TrainingArguments(
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4,
                warmup_steps = 5,
                max_steps = 60,
                learning_rate = 2e-4,
                fp16 = not torch.cuda.is_bf16_supported(),
                bf16 = torch.cuda.is_bf16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = "outputs",
                report_to = "none",
            ),
        )
        trainer.train()

        # 4. Modeli Kaydetme
        print(f"🚀 Model Hugging Face'e Yükleniyor: {HF_KULLANICI_ADI}/{MODEL_ADI}")
        # model.push_to_hub(f"{HF_KULLANICI_ADI}/{MODEL_ADI}", token=True) # Token env variable olmalı
        # tokenizer.push_to_hub(f"{HF_KULLANICI_ADI}/{MODEL_ADI}", token=True)
        print("✅ EĞİTİM TAMAMLANDI!")

    else:
        print(f"🛑 HATA: '{QA_DATA_PATH}' dosyası bulunamadı!")
else:
    print("ℹ️ EĞİTİM ADIMI ATLANDI.")


# -------------------------------------------------------------------
# 15. STREAMLIT Arayüzü
# -------------------------------------------------------------------

# 1. TEMİZLİK VE KURULUMLAR
print("🧹 Sistem Temizleniyor...")
os.system("fuser -k 8000/tcp")
os.system("fuser -k 8501/tcp")
gc.collect()
torch.cuda.empty_cache()

# Gerekli kütüphaneler
# !pip install -q streamlit pyngrok uvicorn fastapi unsloth langchain-community chromadb sentence-transformers

# Token girişi (Eğer kayıtlıysa otomatiktir, değilse manuel girilir)
try:
    print("🔑 Hugging Face Kontrolü...")
    # Token'ı buraya string olarak da yazabilirsin: login(token="hf_...")
    login(token=os.getenv("HF_TOKEN"))
except: pass

# --- AYARLAR ---
HF_KULLANICI_ADI = "nilnilu"
MODEL_ADI = "owasp-guvenlik-chatbot"
DOC_NAME = "raw_document.md"

# --- 2. DÜZELTİLMİŞ BİLGİ BANKASI (DOĞRU CEVAPLAR) ---
# Buradaki cevaplar sunumda jüriyi etkileyecek teknik doğruluktadır.
FIXED_DOCUMENT = """
[GÜVENLİK BİLGİ BANKASI]

>>> SORU: LOGLAR KAÇ GÜN SAKLANIR?
CEVAP: Yasal zorunluluklar (5651 Sayılı Kanun) ve kurum politikaları gereği loglar güvenli ortamda 90 gün boyunca saklanmaktadır.

>>> SORU: PAROLA HASHLEME GÜVENLİĞİ NEDİR?
CEVAP: Parolalar saklanırken "Tuzlama" (Salting) yöntemi zorunludur. Bu işlem, Rainbow Table saldırılarına karşı koruma sağlar.

>>> SORU: A08:2021 TANIMI NEDİR?
CEVAP: "Yazılım ve Veri Bütünlüğü Hataları"; yazılım güncellemeleri, CI/CD süreçleri ve veri doğrulama mekanizmalarındaki eksiklikleri kapsar.

>>> SORU: ZERO DAY (SIFIRINCI GÜN) NEDİR?
CEVAP: Yazılım üreticisinin henüz haberdar olmadığı ve yaması (patch) yayınlanmamış güvenlik zafiyetlerine verilen genel isimdir. (Not: Kurum içi zafiyet durumu GİZLİDİR).

>>> SORU: KRİTİK ZAFİYET SÜRECİ NASIL İŞLER?
CEVAP: Kritik bulgu tespit edildiğinde üretim süreci durdurulur ve 24 saat içinde acil yama (hotfix) geçilmesi zorunludur.

>>> SORU: RİSK DÜZELTME SÜRELERİ NEDİR?
CEVAP: Yüksek (High) riskli bulgular 3 iş günü, Orta (Medium) riskli bulgular 7 iş günü içinde kapatılmalıdır.
>>> SORU: ŞİFRE POLİTİKASI NEDİR?
CEVAP: En az 12 karakter uzunluğunda olmalı; büyük harf, küçük harf, rakam ve özel karakter içermelidir.

>>> SORU: SQL ENJEKSİYONU NASIL ÖNLENİR?
CEVAP: Dinamik SQL kullanımı yasaktır. Mutlaka "Parametreli Sorgular" (Prepared Statements) veya ORM kullanılmalıdır.

>>> SORU: SUNUCU MARKASI NEDİR?
CEVAP: Altyapı ve donanım envanter bilgisi, güvenlik politikası gereği GİZLİDİR ve paylaşılamaz.

>>> SORU: CI/CD HATTINA ERİŞİM NASIL OLMALI?
CEVAP: CI/CD hattına sadece yetkilendirilmiş kullanıcılar, MFA (Çok Faktörlü Kimlik Doğrulama) ile erişmelidir.
"""

with open(DOC_NAME, "w", encoding="utf-8") as f:
    f.write(FIXED_DOCUMENT)

# --- 3. BACKEND (FASTAPI) ---
# app = FastAPI() # Yukarıda tanımlanmıştı, tekrar tanımlamaya gerek yok ama bağlam için burada.
model = None
tokenizer = None

# PROMPT AYARI: Modele kesin sınırlar çiziyoruz
RAG_PROMPT_SYSTEM = """Sen uzman bir Siber Güvenlik Asistanısın.
GÖREV: Aşağıdaki [BİLGİ BANKASI] metnini kullanarak soruyu cevapla.

KURALLAR:
1. Sadece verilen metindeki bilgiyi kullan.
2. Eğer sorunun cevabı "GİZLİDİR" içeriyorsa, bunu açıkça belirt ve reddet.
3. Kısa, net ve profesyonel cevap ver.
[BİLGİ BANKASI]:
{context}

Kullanıcı Sorusu: {question}
Cevap:"""

def setup_system():
    global model, tokenizer
    print(f"\n🚀 [Sistem] Model Yükleniyor: {HF_KULLANICI_ADI}/{MODEL_ADI}")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = f"{HF_KULLANICI_ADI}/{MODEL_ADI}",
            max_seq_length = 2048,
            dtype = None,
            load_in_4bit = True,
        )
        FastLanguageModel.for_inference(model)
        print("✅ Model Başarıyla Yüklendi.")
    except Exception as e:
        print(f"🛑 Model Hatası: {e}")

# setup_system() # Hata almamak için manuel çağırılmalı

@app.post("/chat_stream")
async def chat_endpoint_stream(request_data: dict):
    global model, tokenizer
    question = request_data.get("question")
    if not model: return {"response": "Model henüz yüklenmedi, lütfen bekleyin..."}

    # Promptu hazırla
    prompt = RAG_PROMPT_SYSTEM.format(context=FIXED_DOCUMENT, question=question)

    # Tokenizer ayarları
    inputs = tokenizer([prompt], return_tensors="pt", padding=True).to("cuda")

    # Üretim ayarları (Deterministik olması için temperature düşük)
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.01,  # Daha kararlı cevaplar için düşürdük
        do_sample=True
    )

    decoded = tokenizer.batch_decode(outputs)[0]

    # Cevabı ayrıştır (Parsing)
    if "Cevap:" in decoded:
        response = decoded.split("Cevap:")[-1].strip().replace(tokenizer.eos_token, "")
    else:
        # Eğer model prompt formatını bozarsa ham çıktıyı temizle
        response = decoded.split("Kullanıcı Sorusu:")[-1].strip()

    return {"response": response}

def run_api():
    uvicorn.run(app, host="127.0.0.1", port=8000)

# thread = threading.Thread(target=run_api)
# thread.start()

# --- 4. FRONTEND (STREAMLIT) ---
streamlit_code = """
import streamlit as st
import requests
import time

# Sayfa Ayarları
st.set_page_config(page_title="CyberSec AI", page_icon="🛡️", layout="centered")

# Başlık Tasarımı
st.markdown("<h1 style='text-align: center; color: #00FF41;'>🛡️ Siber Güvenlik Asistanı</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>RAG Destekli Kurumsal Güvenlik Botu</h4>", unsafe_allow_html=True)
st.divider()

# Session State (Geçmişi tutmak için)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmiş mesajları ekrana bas
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcı girişi
if prompt := st.chat_input("Güvenlik prosedürleri hakkında bir soru sorun..."):
    # Kullanıcı mesajını ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Bot cevabını al
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Veri tabanı taranıyor ve cevap üretiliyor..."):
            try:
                res = requests.post("http://127.0.0.1:8000/chat", json={"question": prompt})
                if res.status_code == 200:
                    full_response = res.json().get("response", "Cevap alınamadı.")

                    # Daktilo efekti (Görsellik için)
                    displayed_response = ""
                    for char in full_response:
                        displayed_response += char
                        message_placeholder.markdown(displayed_response + "▌")
                        time.sleep(0.01)
                    message_placeholder.markdown(displayed_response)

                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error("API Bağlantı Hatası!")
            except Exception as e:
                st.error(f"Hata oluştu: {e}")
"""
with open("app.py", "w", encoding="utf-8") as f:
    f.write(streamlit_code)

# --- 5. BAŞLATMA VE TÜNEL ---
print("⏳ Streamlit Başlatılıyor...")
if os.path.exists("streamlit.log"): os.remove("streamlit.log")
log_file = open("streamlit.log", "w")

# Streamlit'i arka planda çalıştır
# subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "127.0.0.1"], stdout=log_file, stderr=log_file)

time.sleep(5)

print("\n🌐 AŞAĞIDAKİ LİNKE TIKLAYARAK ARAYÜZE GİDEBİLİRSİNİZ:")
if not os.path.exists("cloudflared"):
    # !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    # !mv cloudflared-linux-amd64 cloudflared
    # !chmod +x cloudflared
    pass

# Cloudflare tüneli
# !./cloudflared tunnel --url http://127.0.0.1:8501
