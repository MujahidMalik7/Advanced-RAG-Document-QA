from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
import os
from dotenv import load_dotenv

load_dotenv()

def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = Chroma(persist_directory='./chroma_db', embedding_function=embeddings)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    llm = ChatAnthropic(model = "claude-haiku-4-5-20251001")
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    #base retriever
    retriever = MultiQueryRetriever.from_llm(retriever = base_retriever, llm = llm)
    #base compresser
    reranker = CrossEncoderReranker(top_n = 5, model = cross_encoder)
    #final retriever
    final_retriever = ContextualCompressionRetriever(base_retriever = retriever, base_compressor = reranker)

    return final_retriever