import streamlit as st
from retriever import get_retriever
from chain import get_answer

st.title("Advanced Rag-Based System")

retriever = get_retriever()
question = st.text_input("Enter your question:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching documents..."):
            answer, citations = get_answer(question, retriever)
            st.subheader("Answer:")
            st.write(answer)
            st.divider()
            st.subheader("Sources:")
            for citation in citations:
                st.write(f"📄 {citation}")