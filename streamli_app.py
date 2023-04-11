import streamlit as st
from sentence_transformers import SentenceTransformer, util
import json
from os.path import exists
import numpy as np
import math
import gspread
from google.oauth2 import service_account
from io import StringIO
import openai
import pandas as pd
import math
import urllib.request
from gspread_pandas import Spread,Client
import base64
#from gsheetsdb import connect


@st.cache_resource
def download_file():
    url = "https://drive.google.com/uc?export=download&id=1e_bneSaNGhY77Nt07RhTjcMekvwHRGjS"
    path = "file.json"

    # Use urllib.request.urlretrieve to download the file from the given URL
    urllib.request.urlretrieve(url, path)

    # Return the path to the downloaded file
    return path

# Download the file and get the path to the downloaded file
path = download_file()


# Set up the OpenAI API key
openai.api_key = st.secrets["api_secret"]

prompt_text = "You are an summary bot."

# Define the OpenAI function
def openaiapi(input_text):
    messages = [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": input_text}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=messages,
        temperature=0.7,
        max_tokens=2000,
        n=1,
        stop=None,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response


@st.cache(allow_output_mutation=True)
def load_model():
    model1 = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    return model1

model = load_model()

def get_embeddings(texts):
    if type(texts) == str:
        texts = [texts]
    texts = [text.replace("\n", " ") for text in texts]
    return model.encode(texts)
  
def read_json(json_path):
    print('Loading embeddings from "{}"'.format(json_path))
    with open(json_path, 'r') as f:
        values = json.load(f)
    return (values['chapters'], np.array(values['embeddings']))


def read_epub(book_path, json_path, preview_mode, first_chapter, last_chapter):
    chapters = get_chapters(book_path, preview_mode, first_chapter, last_chapter)
    if preview_mode:
        return (chapters, None)
    print('Generating embeddings for chapters {}-{} in "{}"\n'.format(first_chapter, last_chapter, book_path))
    paras = [para for chapter in chapters for para in chapter['paras']]
    embeddings = get_embeddings(paras)
    try:
        with open(json_path, 'w') as f:
            json.dump({'chapters': chapters, 'embeddings': embeddings.tolist()}, f)
    except:
        print('Failed to save embeddings to "{}"'.format(json_path))
    return (chapters, embeddings)

def process_file(path, preview_mode=False, first_chapter=0, last_chapter=math.inf):
    values = None
    if path[-4:] == 'json':
        values = read_json(path)
    elif path[-4:] == 'epub':
        json_path = 'embeddings-{}-{}-{}.json'.format(first_chapter, last_chapter, path)
        if exists(json_path):
            values = read_json(json_path)
        else:
            values = read_epub(path, json_path, preview_mode, first_chapter, last_chapter) 
    else:
        print('Invalid file format. Either upload an epub or a json of book embeddings.')        
    return values
  
chapters, embeddings = process_file(path)
  
def index_to_para_chapter_index(index, chapters):
    for chapter in chapters:
        paras_len = len(chapter['paras'])
        if index < paras_len:
            return chapter['paras'][index], chapter['title'], index
        index -= paras_len
    return None

def search(query, embeddings, n=3):
    query_embedding = get_embeddings(query)[0]
    scores = np.dot(embeddings, query_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding))
    results = sorted([i for i in range(len(embeddings))], key=lambda i: scores[i], reverse=True)[:n]

    search_results = []
    for index in results:
        para, title, para_no = index_to_para_chapter_index(index, chapters)
        search_results.append(f"{para} ({title}, para {para_no})")

    return search_results



st.title("Streamlit App for Ebook Search and OpenAI Integration")
book_podcast_name = st.text_input("A) Input box for a book/podcast name")
#embeddings_link = st.text_input("B) Input for Link to the JSON Embeddings")

initial_questions = st.text_area("C) Input for List of Initial Questiuons (One per Line)").split("\n")
num_follow_up_questions = st.slider("D) Amount of follow-up questions", 1, 10)
submit_button = st.button("Submit")

raw_api_responses = [] 

def append_to_dataframe(df, data):
    data_string = StringIO(data)
    new_df = pd.read_csv(data_string, sep='|')
    return pd.concat([df, new_df], ignore_index=True)

def save_responses_to_file(responses, filename):
    with open(filename, 'w') as f:
        json.dump(responses, f, indent=2)

def get_download_link(filename, text):
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download {filename}</a>'


if submit_button:
    columns = ["Question", "Paragraph/Sentence/Quote", "30 Word Summary", "Tag 1", "Tag 2", "Tag 3", "Tag 4", "Tag 5", "7 Word Problem Summary", "Emotion Triggered", "Counter-Intuitive or Counter-Narrative or Elegant Articulation"]
    results_df = pd.DataFrame(columns=columns)

    for question in initial_questions:
        search_results = search(question, embeddings)
        
        prompt1 = "The below are extracts based on a semantic search from a book or a podcast transcript. \nI want you to extract lessons or principles or secrets for success, building wealth, business advice and/or investing in a table in the following format: \n\n|Question| Paragraph/Sentence/Quote\t|30 Word Summary |Tag 1|Tag 2|Tag 3|Tag 4|Tag 5|7 Word Problem Summary| Emotion Triggered | Counter-Intuitive or Counter-Narrative or Elegant Articulation|\n\nParagraph/Sentence/Quote - This must be an extract from the text. It must be either counter-intuitive (Not how I expected the world to work) or counter-narrative (Not how I was told it works), or be elegantly articulated (wish that I could have said it like that). \n\nEmotion Triggered: | LOL – That’s so funny| WTF – That pisses me off | AWW – That’s so cute | WOW – That’s amazing | NSFW – That’s Crazy| OHHHH – Now I Get it | FINALLY – someone said what I feel| YAY – That’s great news|\n\nExtract:\n"

        search_results_text = "\n".join(search_results)
        prompt1_with_results = f"{prompt1}\n{search_results_text}"

        api_response = openaiapi(prompt1_with_results)
        raw_api_responses.append(api_response.choices[0].to_dict())  # 

        results_df = append_to_dataframe(results_df, api_response)

        prompt2 = f"Think like the best podcast interviewer. What will be the  {num_follow_up_questions} best follow-up questions to ask?\n\nQuestion 1: \n"

        follow_up_api_response = openaiapi(f"{prompt2}\n{api_response}")
        raw_api_responses.append(follow_up_api_response.choices[0].to_dict())

        follow_up_questions = follow_up_api_response.split("\n")
        for follow_up_question in follow_up_questions:
            follow_up_api_response = openaiapi(f"{follow_up_question}\n{search_results_text}")

            results_df = append_to_dataframe(results_df, follow_up_api_response)

    st.success("Task Completed")
    st.write(results_df)
        # ... (existing code) ...

    # Save the raw API responses to a text file
    output_filename = 'raw_api_responses.json'
    save_responses_to_file(raw_api_responses, output_filename)

    # Display the download link
    with open(output_filename, 'r') as f:
        text = f.read()
        download_link = get_download_link(output_filename, text)
        st.markdown(download_link, unsafe_allow_html=True)

    
    
