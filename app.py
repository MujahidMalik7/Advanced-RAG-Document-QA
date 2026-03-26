import streamlit as st
from retriever import get_retriever
from chain import get_answer
from audiorecorder import audiorecorder
from faster_whisper import WhisperModel
import io

st.set_page_config(page_title="Advanced RAG", page_icon="📄")
st.title("Advanced RAG Document QA")

@st.cache_resource
def load_retriever():
    retriever = get_retriever()
    return retriever

@st.cache_resource
def load_whisper():
    model = WhisperModel("small.en", device="cuda", compute_type="float16")
    return model

def display_answer(question, input_ph, output_ph, cost_ph):
    with st.spinner("Searching documents..."):
        answer, citations, cb = get_answer(question, load_retriever())
        
        if cb:
            st.session_state["total_input_tokens"] += cb["input_tokens"]
            st.session_state["total_output_tokens"] += cb["output_tokens"]
            st.session_state["total_cost"] += cb["total_cost"]
            input_ph.metric("Input Tokens", st.session_state["total_input_tokens"])
            output_ph.metric("Output Tokens", st.session_state["total_output_tokens"])
            cost_ph.metric("Total Cost ($)", round(st.session_state["total_cost"], 6))

        st.subheader("Answer:")
        st.write(answer)
        st.divider()
        st.subheader("Sources:")
        for citation in citations:
            st.markdown(f"* 📄 **Source:** {citation}")

if "total_input_tokens" not in st.session_state:
    st.session_state["total_input_tokens"] = 0
if "total_output_tokens" not in st.session_state:
    st.session_state["total_output_tokens"] = 0
if "total_cost" not in st.session_state:
    st.session_state["total_cost"] = 0.0

input_placeholder = st.sidebar.empty()
output_placeholder = st.sidebar.empty()
cost_placeholder = st.sidebar.empty()

input_placeholder.metric("Input Tokens", st.session_state["total_input_tokens"])
output_placeholder.metric("Output Tokens", st.session_state["total_output_tokens"])
cost_placeholder.metric("Total Cost ($)", round(st.session_state["total_cost"], 6))

audio = audiorecorder("Start Recording", "Stop Recording")

if len(audio) > 0 and audio != st.session_state.get("last_audio"):
    st.session_state["last_audio"] = audio
    with st.spinner("Transcribing audio..."):
        model = load_whisper()
        buffer = io.BytesIO()
        audio.export(buffer, format = "wav")
        buffer.seek(0)
        segments, _ = model.transcribe(buffer)
        transcript_text = " ".join([s.text for s in segments])
        st.session_state["question"] = transcript_text
        st.info(f"💬 You asked: {transcript_text}")

question = st.text_input("Enter your question: ", value=st.session_state.get("question", ""))

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        st.info(f"💬 You asked: {question}")
        display_answer(question, input_placeholder, output_placeholder, cost_placeholder)