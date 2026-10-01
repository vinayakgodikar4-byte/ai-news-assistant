from datasets import load_dataset
from transformers import pipeline
from rouge_score import rouge_scorer
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

print("\n📦 Loading dataset (please wait)...")

# Load dataset
dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:1%]")

# Load models once
print("🤖 Loading models...")

summarizer = pipeline("summarization", model="t5-small")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

print("✅ Setup complete!")

# Functions
def get_summary(text):
    text = text[:1000]  # avoid token limit
    return summarizer(text, max_length=120, min_length=30, do_sample=False)[0]['summary_text']

def extractive_summary(text, n=3):
    sentences = sent_tokenize(text)
    return " ".join(sentences[:n])

def evaluate(reference, abs_summary, ext_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores_abs = scorer.score(reference, abs_summary)
    scores_ext = scorer.score(reference, ext_summary)

    print("\n===== ROUGE SCORES =====")
    print("Abstractive:", scores_abs)
    print("Extractive:", scores_ext)

def chatbot(query, article, abs_summary, ext_summary):
    q = query.lower()

    if "summary" in q:
        return abs_summary
    elif "short" in q:
        return ext_summary
    elif "sentiment" in q:
        result = sentiment_model(article[:512])[0]
        return f"Sentiment: {result['label']} (Confidence: {result['score']:.2f})"
    elif "explain" in q:
        return "This news is about: " + abs_summary
    elif "who" in q:
        return "The article involves international organizations and governments."
    elif "impact" in q:
        return "This may have political and legal implications globally."
    else:
        return "Try: summary / short / sentiment / explain / who / impact"

# MAIN LOOP
while True:
    print("\n==============================")
    print("📰 AI NEWS ASSISTANT")
    print("==============================")

    try:
        index = int(input("Enter news index (0-100) or -1 to exit: "))
    except:
        print("❌ Invalid input")
        continue

    if index == -1:
        print("👋 Exiting...")
        break

    if index < 0 or index >= len(dataset):
        print("❌ Invalid index")
        continue

    article = dataset[index]['article']
    reference = dataset[index]['highlights']

    print("\n===== ORIGINAL ARTICLE =====\n")
    print(article[:1000])

    # Generate summaries
    abs_summary = get_summary(article)
    ext_summary = extractive_summary(article)

    print("\n===== ABSTRACTIVE SUMMARY =====\n")
    print(abs_summary)

    print("\n===== EXTRACTIVE SUMMARY =====\n")
    print(ext_summary)

    # Evaluate
    evaluate(reference, abs_summary, ext_summary)

    # Chatbot loop for this article
    print("\n===== CHATBOT =====")
    while True:
        q = input("\nAsk (type 'next' for new article): ")

        if q.lower() == "next":
            break

        print(chatbot(q, article, abs_summary, ext_summary))
