import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from ingest import process_pdfs

# Constantes
VECTORSTORE_PATH = "vectorstore"
OPENAI_MODEL = "gpt-3.5-turbo" # Modelo mais rápido e econômico
# OPENAI_MODEL = "gpt-4" # Modelo mais poderoso

def load_chain():
    """
    Carrega e configura a cadeia de conversação (LangChain).
    """
    # 1. Carregar o banco de dados vetorial
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    vector_store = FAISS.load_local(
        VECTORSTORE_PATH,
        embedding_function,
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever()

    # 2. Definir o template do prompt
    # Este prompt instrui o LLM a usar o contexto fornecido para responder.
    template = """
    Você é um assistente prestativo. Sua tarefa é responder à pergunta do usuário
    baseando-se exclusivamente no seguinte contexto.
    Se a resposta não estiver no contexto, diga "Não encontrei a resposta nos documentos fornecidos.".
    Não invente informações.

    Contexto:
    {context}

    Pergunta:
    {question}

    Resposta:
    """
    prompt = PromptTemplate.from_template(template)

    # 3. Configurar o LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    # 4. Criar a cadeia (chain) usando LangChain Expression Language (LCEL)
    # A cadeia irá:
    # - Recuperar documentos relevantes (contexto)
    # - Passar o contexto e a pergunta para o prompt
    # - Passar o prompt formatado para o LLM
    # - Extrair a resposta do LLM
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def main():
    """
    Função principal que executa a aplicação Streamlit.
    """
    load_dotenv()

    # Configuração da página do Streamlit
    st.set_page_config(page_title="AS05: Assistente de Documentos PDF", page_icon="🤖")
    st.title("🤖 Assistente de Documentos PDF")
    
    # Upload de PDF
    uploaded_file = st.file_uploader("Envie seu arquivo PDF", type="pdf")
    
    if uploaded_file is not None:
        # Verificar se já foi processado nesta sessão
        if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
            # Criar pasta docs se não existir
            os.makedirs("docs", exist_ok=True)
            
            # Limpar pasta docs
            for file in os.listdir("docs"):
                os.remove(os.path.join("docs", file))
            
            # Salvar arquivo enviado
            with open(f"docs/{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Processar PDF automaticamente
            with st.spinner("Processando PDF..."):
                success, message = process_pdfs()
                if success:
                    st.success(f"✅ {message}")
                    st.session_state.processed_file = uploaded_file.name
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
                    st.stop()
    
    # Verificar se o banco de dados vetorial existe
    if not os.path.exists(VECTORSTORE_PATH):
        st.info("📄 Envie um arquivo PDF para começar!")
        st.stop()

    # Verificar se a chave do Google está disponível
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Chave de API do Google não encontrada. Configure GOOGLE_API_KEY nos secrets.")
        st.stop()

    st.divider()

    # Carregar a cadeia de RAG
    try:
        rag_chain = load_chain()
    except Exception as e:
        st.error("Ocorreu um erro ao carregar a cadeia de conversação.")
        st.error(f"Detalhes: {e}")
        st.stop()


    # Inicializar o histórico da conversa
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibir mensagens do histórico
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input do usuário
    if prompt := st.chat_input("Faça uma pergunta sobre seus documentos"):
        # Adicionar mensagem do usuário ao histórico
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Gerar e exibir a resposta do assistente
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                response = rag_chain.invoke(prompt)
                st.markdown(response)
        
        # Adicionar resposta do assistente ao histórico
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()