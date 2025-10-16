import os
import pandas as pd
import streamlit as st
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import tensorflow as tf

# TensorFlow setting
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.compat.v1.reset_default_graph()

# Set NLTK data path for cloud environment
nltk_data_path = './nltk_data'
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

# Function to ensure NLTK resources are downloaded
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords.zip')
    except LookupError:
        nltk.download('stopwords', download_dir=nltk_data_path)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir=nltk_data_path)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', download_dir=nltk_data_path)

# Download necessary NLTK resources
download_nltk_data()

# Load NLP models
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Streamlit UI Setup
st.title("NLP Pipeline with Streamlit")
st.write("This app demonstrates various NLP concepts.")

# Sidebar for options
st.sidebar.title("Options")
task = st.sidebar.selectbox("Choose an NLP task", 
                            ["Word Generation", "Stop Words Removal", "Lemmatization", "N-Grams", 
                             "POS Tagging", "NER", "Text Similarity"])

text_input = st.text_area("Enter Text Here")

# 1. Word Generation
if task == "Word Generation":
    st.subheader("Text Generation using GPT-2")
    if text_input:
        generator = pipeline("text-generation", model="gpt2")
        generated_text = generator(text_input, max_length=5, num_return_sequences=1)[0]['generated_text']
        st.write(generated_text)

# 2. Stop Words Removal
elif task == "Stop Words Removal":
    st.subheader("Stop Words Removal")
    if text_input:
        word_tokens = word_tokenize(text_input)
        filtered_sentence = [word for word in word_tokens if word.lower() not in stop_words]
        st.write(' '.join(filtered_sentence))

# 3. Lemmatization
elif task == "Lemmatization":
    st.subheader("Lemmatization")
    if text_input:
        doc = nlp(text_input)
        lemmatized_sentence = [token.lemma_ for token in doc]
        st.write(' '.join(lemmatized_sentence))

# 4. N-Grams
elif task == "N-Grams":
    st.subheader("N-Gram Extraction")
    n_value = st.sidebar.slider("Select N for N-Grams", 1, 5, 2)
    if text_input:
        vectorizer = CountVectorizer(ngram_range=(n_value, n_value))
        ngrams_matrix = vectorizer.fit_transform([text_input])
        ngrams = vectorizer.get_feature_names_out()
        ngrams_counts = ngrams_matrix.toarray().flatten()
        df = pd.DataFrame({
            'N-Gram': ngrams,
            'Frequency': ngrams_counts
        })
        st.dataframe(df)

# 5. POS Tagging
elif task == "POS Tagging":
    st.subheader("Part-of-Speech Tagging")
    if text_input:
        doc = nlp(text_input)
        pos_tags = [(token.text, token.pos_) for token in doc]
        pos_df = pd.DataFrame(pos_tags, columns=["Word", "POS Tag"])
        st.dataframe(pos_df)

# 6. Named Entity Recognition (NER)
elif task == "NER":
    st.subheader("Named Entity Recognition")
    if text_input:
        doc = nlp(text_input)
        entities = [(entity.text, entity.label_) for entity in doc.ents]
        df_entities = pd.DataFrame(entities, columns=["Entity", "Label"])
        st.dataframe(df_entities, use_container_width=True)

# 7. Text Similarity Recognizer (TSR)
elif task == "Text Similarity":
    st.subheader("Text Similarity")
    text_input_2 = st.text_area("Enter Second Text Here")
    if text_input and text_input_2:
        vectorizer = CountVectorizer().fit_transform([text_input, text_input_2])
        vectors = vectorizer.toarray()
        cosine_sim = cosine_similarity(vectors)
        st.write(f"Cosine Similarity: {cosine_sim[0][1]}")
