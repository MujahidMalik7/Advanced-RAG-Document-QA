from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def ingest():
    loader = PyPDFDirectoryLoader('./data')
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    docsearch = Chroma.from_documents(docs, embeddings, persist_directory = './chroma_db')

    print (f"Ingested {len(docs)} chunks into ChromaDB")

if __name__ == "__main__":
    ingest()