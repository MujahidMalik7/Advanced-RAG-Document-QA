import streamlit as st
from retriever import get_retriever
from chain import get_answer

st.set_page_config(page_title="Advanced RAG", page_icon="📄")
st.title("Advanced RAG Document QA")

@st.cache_resource
def load_retriever():
    retriever = get_retriever()
    return retriever

question = st.text_input("Enter your question:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching documents..."):
            answer, citations = get_answer(question, load_retriever())
            st.subheader("Answer:")
            st.write(answer)
            st.divider()
            st.subheader("Sources:")
            for citation in citations:
                st.markdown(f"* 📄 **Source:** {citation}")