import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Import Modul Modern LangChain Core & Google GenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(
    title="SARTHAKA AI Board Game Advisor API",
    version="1.0.0"
)

# Mengaktifkan CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Inisialisasi awal sebagai None untuk Mencegah NameError / Crash Sejak Awal
retriever = None
rag_chain = None

# 2. Ambil Model Embedding & LLM Gemini
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

# 3. Struktur Prompt Advisor Finansial Game
system_prompt = (
    "Anda adalah AI Advisor Finansial senior dan Asisten Aturan resmi untuk board game 'SARTHAKA'.\n"
    "Tugas Anda adalah menjawab pertanyaan seputar mekanisme game dan memberikan analisis risiko keuangan "
    "kepada pemain berdasarkan aturan resmi permainan.\n\n"
    "Konteks Aturan SARTHAKA:\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 4. Proses Pemuatan Safe-Load (Penyelamat dari Crash)
try:
    # Membaca folder index FAISS
    vector_store = FAISS.load_local(
        "faiss_sarthaka_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # RAG Chain HANYA dirakit jika retriever berhasil dibuat tanpa error
    rag_chain = (
        {
            "context": retriever | format_docs, 
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    print("✅ DATABASE VEKTOR & RAG CHAIN BERHASIL DIMUAT LENGKAP!")

except Exception as e:
    print("\n" + "="*60)
    print("❌ AMAN: GAGAL MEMUAT DATABASE VEKTOR FAISS!")
    print(f"Detail Error: {e}")
    print("Penyebab Umum: File index.faiss atau index.pkl berukuran 0 bytes (kosong).")
    print("Efek: Server tetap menyala, tetapi endpoint /api/advisor belum bisa digunakan.")
    print("="*60 + "\n")


# 5. Model Data Pydantic untuk Request
class PlayerState(BaseModel):
    profesi: str
    level_sekarang: int
    gabah_tunai: int
    utang_leuit: int

class AdvisorRequest(BaseModel):
    question: str
    player_state: Optional[PlayerState] = None


# 6. Endpoint Utama dengan Proteksi Status Chain
@app.post("/api/advisor")
async def consult_advisor(request: AdvisorRequest):
    # Proteksi: Jika rag_chain masih berstatus None, cegah eksekusi dan kirim respons error 500
    if rag_chain is None:
        raise HTTPException(
            status_code=500, 
            detail="Server Backend menyala, tetapi fitur AI tidak dapat diakses karena file index FAISS "
                   "di dalam folder 'faiss_sarthaka_index/' kosong (0 bytes) atau rusak. "
                   "Silakan ganti dengan file index yang valid dari Google Colab."
        )
    
    try:
        formatted_input = f"Pertanyaan Pemain: {request.question}\n\n"
        if request.player_state:
            state = request.player_state
            formatted_input += (
                f"KONDISI FINANSIAL PEMAIN SAAT INI:\n"
                f"- Profesi: {state.profesi}\n"
                f"- Level Perkembangan: Level {state.level_sekarang}\n"
                f"- Gabah Tunai: {state.gabah_tunai} Gabah\n"
                f"- Utang di Leuit: {state.utang_leuit} Gabah\n"
            )
        
        answer = rag_chain.invoke(formatted_input)
        return {"status": "success", "answer": answer}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {
        "status": "online",
        "faiss_status": "READY" if rag_chain is not None else "ERROR_0_BYTES_DETECTED",
        "message": "SARTHAKA Backend API is running."
    }