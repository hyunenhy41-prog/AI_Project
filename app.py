import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Page Configuration
st.set_page_config(page_title="AI Review Sentiment Analyzer", layout="wide")

@st.cache_resource
def load_and_train_model():
    # 1. Load Data
    try:
        df = pd.read_csv('amazon_cells_labelled.txt', sep='\t', header=None, names=['review', 'label'])
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'amazon_cells_labelled.txt' is in the directory.")
        return None, None, None

    # 2. Text Preprocessing Function
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text

    df['cleaned_review'] = df['review'].apply(clean_text)

    # 3. Train-Test Split and Pipeline Construction
    X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['label'], test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    
    # 4. Internal Model Evaluation
    X_test_vec = vectorizer.transform(X_test)
    accuracy = accuracy_score(y_test, model.predict(X_test_vec))
    
    return model, vectorizer, accuracy

# Train the model (Cached to run only once)
model, vectorizer, model_accuracy = load_and_train_model()

# Frontend UI Design
st.title("E-commerce Review Sentiment Analyzer")
st.markdown("Enter a customer product review to classify its sentiment and generate business insights.")

if model_accuracy:
    st.sidebar.info(f" Current AI Model Accuracy (Test Set): **{model_accuracy*100:.1f}%**")

user_input = st.text_area("Enter a customer review in English:", height=100, 
                          placeholder="e.g., The battery life is amazing and the display is gorgeous!")

if st.button("Analyze Review"):
    if user_input:
        # Preprocess and Vectorize User Input
        clean_input = user_input.lower()
        clean_input = re.sub(r'[^a-z\s]', '', clean_input)
        input_vec = vectorizer.transform([clean_input])
        
        # Perform Prediction
        prediction = model.predict(input_vec)[0]
        probabilities = model.predict_proba(input_vec)[0]
        confidence = max(probabilities) * 100
        
        # Extract Core Keywords (Based on TF-IDF Scores)
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = input_vec.toarray()[0]
        keywords = [feature_names[i] for i in tfidf_scores.argsort()[-3:][::-1] if tfidf_scores[i] > 0]
        
        st.divider()
        st.subheader("Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        
        # Display Results
        if prediction == 1:
            col1.success("**Positive**")
            col2.metric("Confidence Level", f"{confidence:.1f}%")
            col3.write(f"**Core Keywords:** {', '.join(keywords) if keywords else 'None extracted'}")
            
            st.info("**Business Insight:** This is highly positive feedback. It is recommended to highlight the extracted core keywords in your product detail pages or marketing copywriting.")
        else:
            col1.error("**Negative**")
            col2.metric("Confidence Level", f"{confidence:.1f}%")
            col3.write(f"**Core Keywords:** {', '.join(keywords) if keywords else 'None extracted'}")
            
            st.warning("**Business Insight:** A complaint has been detected. The CS department should contact the customer immediately. Forward the core keywords to the product development team to request improvements.")
    else:
        st.warning("Please enter a review text to analyze.")