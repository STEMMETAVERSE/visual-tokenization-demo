import streamlit as st
from transformers import AutoTokenizer
@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer = load_tokenizer()
st.title("🔤 Visual Tokenization Demo")
text = st.text_area("Enter text")
if st.button("Tokenize"):
    tokens = tokenizer.tokenize(text)
    st.subheader("Tokens")
    st.write(tokens)
    st.subheader("Token Count")
    st.write(len(tokens))
