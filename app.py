import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import gradio as gr

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GOOGLE_API_KEY)

def load_docs(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)
  
def test_pdf_text(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print("\n\n📄 PDF'den okunan sayfa sayısı:", len(pages))
    for i, page in enumerate(pages):
        print(f"\n--- Sayfa {i+1} ---\n", page.page_content[:500])  # sadece ilk 500 karakter

def create_qa_chain(docs):
    vectordb = FAISS.from_documents(docs, embedding_model)
    retriever = vectordb.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def ask_question(file, question):
    try:
        docs = load_docs(file.name)
        if not docs:
            return "❌ PDF içeriği boş veya okunamadı."
        
        print(f"📑 {len(docs)} parça belge üretildi.")
        qa_chain = create_qa_chain(docs)
        result = qa_chain.run(question)
        return result
    except Exception as e:
        print(f"❌ Hata: {str(e)}")
        return f"❌ Hata: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("## 📄 Dökümana Soru Sor (LangChain + Gemini Flash)")
    file = gr.File(label="PDF Yükle", file_types=[".pdf"])
    question = gr.Textbox(label="Sorunuz", placeholder="Bu döküman ne anlatıyor?")
    answer = gr.Textbox(label="Cevap", lines=5)
    btn = gr.Button("Sor")
    
    btn.click(fn=ask_question, inputs=[file, question], outputs=answer)

demo.launch()
