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
        return "I couldn't find relevant information in the documents.", [], None

    context = "\n\n".join([doc.page_content for doc in docs])
    
    citations = list(set([f"{doc.metadata['source']} page {doc.metadata['page']}" for doc in docs]))

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer based on the provided context only. Keep answers concise."),
        ("human", "Question: {question}\n\nContext: {context}")
    ])

    response = (prompt | model).invoke({"question": question, "context": context})
    answer = StrOutputParser().invoke(response)
    usage = response.usage_metadata
    
    cost = (usage["input_tokens"] * 0.0000008) + (usage["output_tokens"] * 0.000004)
    return answer,citations,{"input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"], "total_cost": cost}