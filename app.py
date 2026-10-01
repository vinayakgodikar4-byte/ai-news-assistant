import streamlit as st
from datasets import load_dataset
from transformers import pipeline
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
import nltk

# ---------------- SETUP ----------------
nltk.download('punkt')

st.set_page_config(page_title="AI News Assistant", layout="wide")

# ---------------- PREMIUM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

.block-container {
    max-width: 1100px;
    margin: auto;
}

h1 {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #7c3aed, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.glass {
    background: rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    margin-bottom: 20px;
    transition: 0.3s;
}
.glass:hover {
    transform: translateY(-3px);
}

.stButton>button {
    background: linear-gradient(90deg,#7c3aed,#06b6d4);
    border-radius: 12px;
    padding: 10px 18px;
    font-weight: 600;
    border: none;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.05);
}

section[data-testid="stSidebar"] {
    background: #020617;
}

hr {
    border: 1px solid #1e293b;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h1>🧠 AI News Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#94a3b8;'>Summarize • Analyze • Chat with News</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return load_dataset("cnn_dailymail", "3.0.0", split="test[:3%]")

dataset = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.title("📰 News Selector")
index = st.sidebar.slider("Select Article", 0, len(dataset)-1, 0)

article = dataset[index]['article']
reference = dataset[index]['highlights']

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="t5-small")
    sentiment_model = pipeline("sentiment-analysis")
    return summarizer, sentiment_model

summarizer, sentiment_model = load_models()

# ---------------- ARTICLE ----------------
st.markdown("### 📰 Original Article")
st.markdown(f"<div class='glass'>{article[:1500]}...</div>", unsafe_allow_html=True)

# ---------------- ACTION BUTTONS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("✨ AI Summary"):
        st.session_state["abs"] = summarizer(article[:512], max_length=120, min_length=30)[0]['summary_text']

with col2:
    if st.button("📄 Extractive"):
        st.session_state["ext"] = " ".join(sent_tokenize(article)[:3])

with col3:
    if st.button("😊 Sentiment"):
        st.session_state["sent"] = sentiment_model(article[:512])[0]

# ---------------- OUTPUT ----------------
if "abs" in st.session_state:
    st.markdown("### ✨ Abstractive Summary")
    st.markdown(f"<div class='glass'>{st.session_state['abs']}</div>", unsafe_allow_html=True)

if "ext" in st.session_state:
    st.markdown("### 📄 Extractive Summary")
    st.markdown(f"<div class='glass'>{st.session_state['ext']}</div>", unsafe_allow_html=True)

if "sent" in st.session_state:
    st.markdown("### 😊 Sentiment")
    s = st.session_state["sent"]
    st.metric("Sentiment", s['label'], f"{round(s['score']*100,2)}% confidence")

# ---------------- ROUGE ----------------
if st.button("📊 Evaluate ROUGE"):
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)

    abs_summary = summarizer(article[:512])[0]['summary_text']
    ext_summary = " ".join(sent_tokenize(article)[:3])

    abs_score = scorer.score(reference, abs_summary)
    ext_score = scorer.score(reference, ext_summary)

    st.markdown("### 📊 ROUGE Scores")
    st.markdown(f"<div class='glass'>Abstractive: {abs_score}<br><br>Extractive: {ext_score}</div>", unsafe_allow_html=True)

# ---------------- CHATBOT ----------------
st.markdown("## 💬 Chat with News")

if "chat" not in st.session_state:
    st.session_state.chat = []

def chatbot(q):
    q = q.lower()
    if "summary" in q:
        return summarizer(article[:512])[0]['summary_text']
    elif "short" in q:
        return " ".join(sent_tokenize(article)[:2])
    elif "sentiment" in q:
        return str(sentiment_model(article[:512])[0])
    elif "who" in q:
        return "Key entities: Palestinian Authority, ICC, Israel, USA"
    elif "impact" in q:
        return "This may trigger international legal actions and geopolitical tension."
    elif "why" in q:
        return "Because Palestine seeks legal recognition via ICC."
    else:
        return "Try: summary / short / sentiment / who / impact / why"

query = st.chat_input("Ask something about the news...")

if query:
    response = chatbot(query)
    st.session_state.chat.append(("user", query))
    st.session_state.chat.append(("bot", response))

# Display chat
for role, msg in st.session_state.chat:
    with st.chat_message("user" if role=="user" else "assistant"):
        st.write(msg)
