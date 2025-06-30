import os
import shutil
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Constantes
PDFS_PATH = "docs"
VECTORSTORE_PATH = "vectorstore"

def process_pdfs():
    """
    Função para carregar, dividir e vetorizar documentos PDF.
    """
    load_dotenv()
    
    # 1. Carregar os documentos PDF da pasta 'docs'
    if not os.path.exists(PDFS_PATH) or not os.listdir(PDFS_PATH):
        return False, "Pasta 'docs' não existe ou está vazia."

    loader = PyPDFDirectoryLoader(PDFS_PATH)
    documents = loader.load()
    if not documents:
        return False, "Nenhum documento PDF foi carregado."

    # 2. Dividir os documentos em pedaços (chunks)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)

    # 3. Gerar embeddings usando Google
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    # Criar novo banco vetorial
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_function
    )
    
    # Salvar o banco vetorial
    vector_store.save_local(VECTORSTORE_PATH)

    return True, f"Processados {len(documents)} documentos em {len(chunks)} chunks."

def main():
    """
    Função principal para uso via linha de comando.
    """
    print("Iniciando a ingestão de dados...")
    success, message = process_pdfs()
    if success:
        print("✅ Processo de ingestão concluído com sucesso!")
        print(message)
    else:
        print(f"❌ Erro: {message}")

if __name__ == "__main__":
    main()