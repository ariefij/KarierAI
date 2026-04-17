# Tutorial: Deploying Agent HR (FastAPI) to Google Cloud Platform

Panduan ini akan membimbing Anda langkah-demi-langkah untuk mendeploy aplikasi FastAPI (Agent HR) ke **Google Cloud Run** menggunakan **Cloud Build**.

---

## 1. Persiapan Awal

Pastikan Anda sudah memiliki:
1. Akun **Google Cloud Platform** (GCP) dengan tagihan aktif.
2. [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) terinstal di komputer Anda.
3. Struktur project yang sudah memiliki `Dockerfile`.

---

## 2. Inisialisasi Project & API

Jalankan perintah ini di terminal untuk login dan menyiapkan project:

```bash
# Login ke akun Google Anda
gcloud auth login

# Lihat daftar project yang Anda miliki
gcloud projects list

# Set project ID yang ingin digunakan
gcloud config set project finpro-493407

# Aktifkan API yang dibutuhkan
gcloud services enable run.googleapis.com \
    containerregistry.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    cloudresourcemanager.googleapis.com
```

---

## 3. Konfigurasi Izin (IAM) - PROAKTIF

GCP memiliki kebijakan keamanan ketat. Agar deployment tidak error, kita harus memberikan izin khusus kepada **Default Compute Service Account** (yang menjalankan build Anda).

**Jalankan blok perintah ini sekaligus:**

```bash
# Dapatkan Project ID dan Project Number
PROJECT_ID=finpro-493407
PROJECT_NUMBER=94453605119 
$PROJECT_ID --format="value(projectNumber)")

# Berikan semua akses yang dibutuhkan
for ROLE in "roles/storage.objectAdmin" \
            "roles/artifactregistry.admin" \
            "roles/artifactregistry.createOnPushRepoAdmin" \
            "roles/artifactregistry.repoAdmin" \
            "roles/logging.admin"; do
    gcloud projects add-iam-policy-binding finpro-493407 \
        --member="serviceAccount:94453605119-compute@developer.gserviceaccount.com" \
        --role="$ROLE"
done
```

---

## 4. Persiapkan Dockerfile (Port 8080)

Cloud Run mewajibkan aplikasi mendengarkan (listen) pada port **8080**. Pastikan `Dockerfile` Anda menggunakan variabel environment `$PORT`.

Contoh `CMD` di `Dockerfile`:
```dockerfile
# Cloud Run akan menyuntikkan nilai ke variabel PORT
CMD ["sh", "-c", "uvicorn src.agent_st.server:app --host 0.0.0.0 --port ${PORT:-8080}"]
```

---

## 5. Build & Push ke Registry

Gunakan Cloud Build untuk membungkus aplikasi Anda menjadi sebuah container image.

```bash
gcloud builds submit --tag gcr.io/finpro-493407/karier
```

---

## 6. Deploy ke Cloud Run

Kita akan mendeploy image tersebut dan memasukkan semua konfigurasi dari file `.env` secara otomatis.

```bash
# 1. Ekstrak variabel .env menjadi satu baris (format key1=val1,key2=val2)
ENV_VARS=$(grep -v '^#' .env | xargs | sed 's/ /,/g')

# 2. Deploy ke Cloud Run
gcloud run deploy karier\
  --image gcr.io/finpro-493407/karier\
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="$ENV_VARS"
```

---

## 7. Verifikasi

Setelah selesai, GCP akan memberikan **Service URL**. 
- Buka link tersebut untuk mengecek root API.
- Tambahkan `/docs` (misal: `https://agent-resume-xxx.run.app/docs`) untuk mencoba Swagger UI aplikasi Anda.

---

## Troubleshooting Cepat

*   **Error 403 / Permission Denied**: Jalankan ulang langkah **Nomor 3**. Pastikan Anda adalah **Owner** dari project tersebut.
*   **Container Failed to Start**: Pastikan aplikasi Anda tidak di-hardcode ke port 8000. Harus fleksibel mengikuti variabel `$PORT`.
*   **.env tidak terbaca**: Pastikan file `.env` ada di root folder tempat Anda menjalankan perintah deploy.