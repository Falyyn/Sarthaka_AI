FROM python:3.11-slim

# Mengatur direktori kerja
WORKDIR /app

# Menyalin file dependencies
COPY requirements.txt .

# Menginstal pustaka yang dibutuhkan
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh kode (termasuk folder faiss_sarthaka_index)
COPY . .

# Mengekspos port 7860 (Port wajib untuk Hugging Face Spaces)
EXPOSE 7860

# Menjalankan aplikasi FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
