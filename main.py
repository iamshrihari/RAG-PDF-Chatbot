from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from utils import load_pdf, split_docs
from config import EMBEDDING_MODEL

# Note: RetrievalQA is not available in LangChain 1.x
# See comments below for migration path

PDF_PATH = "data/your_pdf.pdf"

# 1. Load and split PDF
docs = load_pdf(PDF_PATH)
chunks = split_docs(docs)

# 2. Create embeddings
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vector_store = FAISS.from_documents(chunks, embeddings)

# 3. Create Retrieval QA Chain
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
# RetrievalQA is not available in LangChain 1.x
# Use this pattern instead:
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# qa_chain = create_stuff_documents_chain(
#     ChatOpenAI(temperature=0),
#     ChatPromptTemplate.from_messages([
#         ("system", "Answer based on context: {context}"),
#         ("human", "{input}")
#     ])
# )
# See https://python.langchain.com/docs/modules/chains/ for migration guide

# 4. Ask questions
# while True:
#     query = input("\nAsk a question (or type 'exit'): ")
#     if query.lower() == "exit":
#         break
#     answer = qa_chain.run(query)
#     print("\nAnswer:", answer)
