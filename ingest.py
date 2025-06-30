import os
import shutil
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

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

    # 3. Gerar embeddings e armazenar no ChromaDB
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Remover banco anterior se existir
    if os.path.exists(VECTORSTORE_PATH):
        try:
            shutil.rmtree(VECTORSTORE_PATH)
        except:
            pass

    # Criar diretório se não existir
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)

    # Criar novo banco vetorial
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=VECTORSTORE_PATH
    )

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