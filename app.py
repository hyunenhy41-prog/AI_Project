import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import google.generativeai as genai

st.set_page_config(page_title="AI Review Sentiment Analysis System", layout="wide")

@st.cache_resource
def load_and_train_model():
    try:
        df = pd.read_csv('amazon_cells_labelled.txt', sep='\t', header=None, names=['review', 'label'])
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'amazon_cells_labelled.txt' is in the directory.")
        return None, None, None

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text

    df['cleaned_review'] = df['review'].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['label'], test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    
    X_test_vec = vectorizer.transform(X_test)
    accuracy = accuracy_score(y_test, model.predict(X_test_vec))
    
    return model, vectorizer, accuracy

model, vectorizer, model_accuracy = load_and_train_model()

st.sidebar.header("⚙️ System Configuration")
api_key = st.sidebar.text_input("Google Gemini API Key:", type="password", help="Required for generating AI business insights.")
if model_accuracy:
    st.sidebar.info(f"💡 Model Accuracy: **{model_accuracy*100:.1f}%**")

st.title("🛍️ E-commerce Review Sentiment Analyzer")
st.markdown("Input a customer product review to classify sentiment and receive strategic business insights powered by Generative AI (LLM).")

user_input = st.text_area("Enter a customer review in English:", height=100, 
                          placeholder="e.g., The battery life is not good")

if st.button("Analyze Review"):
    if user_input:
        clean_input = user_input.lower()
        clean_input = re.sub(r'[^a-z\s]', '', clean_input)
        input_vec = vectorizer.transform([clean_input])
        
        prediction = model.predict(input_vec)[0]
        probabilities = model.predict_proba(input_vec)[0]
        confidence = max(probabilities) * 100
        
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = input_vec.toarray()[0]
        keywords = [feature_names[i] for i in tfidf_scores.argsort()[-3:][::-1] if tfidf_scores[i] > 0]
        
        st.divider()
        st.subheader("📊 Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        
        if prediction == 1:
            col1.success("✨ **Positive**")
            col2.metric("Confidence Level", f"{confidence:.1f}%")
            col3.write(f"**Core Keywords:** {', '.join(keywords) if keywords else 'None extracted'}")
            st.info("📈 **Business Insight:** Highly positive feedback. It is recommended to emphasize the extracted core keywords in your marketing copywriting and product feature highlights.")
            
        else:
            col1.error("🚨 **Negative**")
            col2.metric("Confidence Level", f"{confidence:.1f}%")
            col3.write(f"**Core Keywords:** {', '.join(keywords) if keywords else 'None extracted'}")
            
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    llm_model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    with st.spinner("🤖 AI is generating an immediate business action plan..."):
                        prompt = f"A customer left a negative review: '{user_input}'. The extracted complaint keywords are {keywords}. As an e-commerce management consultant, provide a concise, professional action plan in English (maximum 3 sentences) for the CS and product development teams to resolve this issue and improve customer satisfaction."
                        response = llm_model.generate_content(prompt)
                        
                    st.warning(f"📉 **Business Insight (AI Analysis):**\n\n{response.text}")
                except Exception as e:
                    st.error("The API key is invalid or an error occurred. Please check your credentials.")
            else:
                st.warning("📉 **Business Insight:** A complaint has been detected. Please provide a Google Gemini API key in the sidebar to receive tailored AI-driven strategic action plans.")
    else:
        st.warning("Please enter a review text to analyze.")
