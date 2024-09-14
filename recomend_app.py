import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
from googletrans import Translator
from google.cloud import texttospeech

# Base URL and API key for AIML API
BASE_URL = 'https://api.aimlapi.com'
API_KEY = ''

# Google Custom Search API details
GOOGLE_API_KEY = ""  # Replace with your API key
GOOGLE_CX = ""  # Replace with your custom search engine ID

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Set background color to white */
    body {
        background: white;
    }
    
    /* Style for buttons */
    .stButton>button {
        background-color: none;
        color: white;
        border-radius: 6px;
        padding: 5px;
        font-size: 16px;
        font-weight: bold;
        transition: background-color 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 2px 4px 10px rgba(0, 0, 0, 0.2);
    }
    
    /* Hover effect for buttons */
    .stButton>button:hover {
        background-color: #0096c7;
        color: white;
        box-shadow: 4px 8px 16px rgba(0, 0, 0, 0.2);
    }
    /* Shadows for other elements like headers */
    .stTextInput, .stSelectbox {
        box-shadow: 2px 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Padding for the main app content */
    .main {
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load CSV file, drop NA values
@st.cache_data
def load_data():
    df = pd.read_csv('new.csv', on_bad_lines='skip', engine='python')
    return df.dropna()

df1 = load_data()

# TF-IDF Vectorizer for job descriptions (used for initial recommendations)
@st.cache_resource
def create_similarity_matrix(df):
    tdif = TfidfVectorizer(stop_words='english')
    df['jobdescription'] = df['jobdescription'].fillna('')
    tdif_matrix = tdif.fit_transform(df['jobdescription'])
    return sigmoid_kernel(tdif_matrix, tdif_matrix)

cosine_sim = create_similarity_matrix(df1)

# Create a series of job titles indexed by the title for recommendations
indices = pd.Series(df1.index, index=df1['jobtitle']).drop_duplicates()

# Function to get recommendations based on job title
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:16]  # Recommend 15 jobs
    job_indices = [i[0] for i in sim_scores]
    return df1['jobtitle'].iloc[job_indices].tolist()

# Function to summarize job description using LLaMA API from AIML
def summarize_job_description_llama(job_title):
    job_description = df1[df1['jobtitle'] == job_title]['jobdescription'].iloc[0]
    endpoint = f"{BASE_URL}/summarize"
    headers = {'Authorization': f'Bearer {API_KEY}'}
    data = {
        'text': job_description,
        'model': 'llama-3.1'  # LLaMA model
    }
    response = requests.post(endpoint, json=data, headers=headers)
    if response.status_code == 200:
        return response.json().get('summary', 'Summary not available.')
    else:
        return "Error: Unable to summarize the job description."

# Translate job description using Google Translator API
def translate_description(text, target_language):
    translator = Translator()
    return translator.translate(text, dest=target_language).text

# Search more about the job using Google Custom Search API
def google_search(query):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={GOOGLE_CX}"
    response = requests.get(url)
    return response.json()

# Text-to-Speech using Google Cloud API
def text_to_speech(text):
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(input_text, voice, audio_config)
    
    with open("output.mp3", "wb") as out:
        out.write(response.audio_content)
        st.audio("output.mp3")

# Function to search jobs based on user's skills and location using Google Custom Search API
def google_job_search(query, user_location):
    search_query = f"{query} jobs {user_location}"
    url = f"https://www.googleapis.com/customsearch/v1?q={search_query}&key={GOOGLE_API_KEY}&cx={GOOGLE_CX}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('items', [])
    else:
        return []

# Streamlit UI for recommendations
st.header('Tech Jobs Recommender')

# Job title dropdown
job_list = df1['jobtitle'].unique()
selected_job = st.selectbox("Type or select a job from the dropdown", job_list)

# Button to show recommendations
if st.button('Show Recommendation'):
    recommended_job_names = get_recommendations(selected_job)
    for job in recommended_job_names:
        st.subheader(job)

# Sidebar for personalized recommendations using Google Custom Search API
st.sidebar.header('Personalize Your Search')
user_skills = st.sidebar.text_input("Enter your skills (comma-separated)")
user_location = st.sidebar.text_input("Enter your location")

# Button to get personalized job recommendations based on user input
if st.sidebar.button('Get Personalized Recommendations'):
    if user_skills or user_location:
        search_query = f"{user_skills} {user_location}"
        personalized_job_results = google_job_search(search_query, user_location)

        if personalized_job_results:
            st.subheader(f"Jobs matching your skills '{user_skills}' in '{user_location}':")
            for result in personalized_job_results:
                st.write(f"[{result['title']}]({result['link']})")
        else:
            st.write("No job results found for your input. Please try different keywords.")
    else:
        st.write("Please enter skills and/or location to get personalized job recommendations.")

# Button to summarize job description using LLaMA API
# Button to translate job description
language = st.selectbox("Select Language", ['en', 'es', 'fr', 'de', 'ar'])
if st.button('Translate Job Description'):
    job_description = df1[df1['jobtitle'] == selected_job]['jobdescription'].iloc[0]
    translated_text = translate_description(job_description, language)
    st.write(translated_text)

# Feedback option
feedback = st.radio("Was this recommendation helpful?", ('Yes', 'No'))
if feedback == 'Yes':
    st.success("Thanks for your feedback!")
else:
    st.warning("We'll work to improve our recommendations.")
