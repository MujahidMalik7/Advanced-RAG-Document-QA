from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()

def get_answer(question, retriever):
    
    model = ChatAnthropic(
        model_name = "claude-haiku-4-5-20251001",
    )
    
    docs = retriever.invoke(question)

    if not docs:
        return "I couldn't find relevant information in the documents."

    context = "\n\n".join([doc.page_content for doc in docs])
    
    citations = list(set([f"{doc.metadata['source']} page {doc.metadata['page']}" for doc in docs]))

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer based on the provided context only. Keep answers concise."),
        ("human", "Question: {question}\n\nContext: {context}")
    ])

    chain = prompt | model | StrOutputParser()

    answer = chain.invoke({"question": question, "context": context})

    return answer,citations