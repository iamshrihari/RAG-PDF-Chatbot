import streamlit as st
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.llms import Ollama
from utils import load_pdf, split_docs
from config import EMBEDDING_MODEL

st.title("📄 RAG PDF Chatbot")

uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

if uploaded_file:
    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # Load and split PDF
    docs = load_pdf(pdf_path)
    chunks = split_docs(docs)

    # Create embeddings and FAISS vector store using HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Setup RAG chain using LangChain 1.x Runnables API
    # Define the prompt template
    prompt_template = ChatPromptTemplate.from_template(
        """Answer the question based on the following context:
        
Context:
{context}

Question: {question}

Answer:"""
    )
    
    # Create a runnable chain that formats retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Initialize Ollama LLM (make sure Ollama is running locally)
    try:
        # Use the Ollama model you pulled locally (e.g., 'mistral')
        llm = Ollama(model="mistral")
        use_llm = True
    except Exception as e:
        st.warning(f"⚠️ Ollama not available: {str(e)}\nMake sure Ollama is running. Download from https://ollama.ai")
        use_llm = False
    
    # Build the RAG chain with LLM
    if use_llm:
        rag_chain = (
            {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
            | prompt_template
            | llm
        )
    
    query = st.text_input("Ask a question:")
    if query:
        with st.spinner("Searching and generating answer..."):
            # Get retrieved documents
            retrieved_docs = retriever.invoke(query)
            
            if retrieved_docs:
                st.markdown("### Retrieved Context:")
                for i, doc in enumerate(retrieved_docs, 1):
                    st.markdown(f"**Document {i}:**")
                    st.markdown(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                    st.markdown("---")
                
                # Generate answer using RAG chain
                st.markdown("### Answer:")
                if use_llm:
                    try:
                        answer = rag_chain.invoke(query)
                        st.success(answer)
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
                else:
                    st.info("⚠️ LLM not available. Configure Ollama or another LLM to generate answers.")
            else:
                st.warning("No relevant documents found.")
