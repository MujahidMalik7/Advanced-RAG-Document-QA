from retriever import get_retriever
from chain import get_answer
import os
from dotenv import load_dotenv
load_dotenv()

retriever = get_retriever()

while True:
    question = input("\n\nEnter a question: ")
    
    if not question.strip():
        continue
    
    if question.lower() in ["exit", "quit", "x"]:
        break

    answer, citations = get_answer(question, retriever)
    print ("\n")
    print("Answer:", answer)
    print("Citations:", citations)