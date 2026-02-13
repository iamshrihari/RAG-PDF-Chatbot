from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(path):
    """
    Load PDF and return list of LangChain Document objects
    """
    loader = PyPDFLoader(path)
    documents = loader.load()
    return documents

def split_docs(documents, chunk_size=500, chunk_overlap=50):
    """
    Split documents into chunks for embeddings
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks
