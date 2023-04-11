import streamlit as st
from sentence_transformers import SentenceTransformer, util
import json
from os.path import exists
import numpy as np
import math
import openai
import pandas as pd
import math
import urllib.request

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
    response = openai.Completion.create(
        engine="davinci",
        prompt="\n".join([m["content"] for m in messages]),
        temperature=0.7,
        max_tokens=2000,
        n=1,
        stop=None,
        frequency_penalty=0,
        presence_penalty=0
    )
    answer = response.choices[0].text.strip()
    return answer


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

if submit_button:
    data = []
    for question in initial_questions:
        # Search the ebook
        search_results = search(question, embeddings)
        # Add search results to table data
        for result in search_results:
            data.append({
                'Question': question,
                'Paragraph/Sentence/Quote': result,
                '30 Word Summary': '',
                'Tag 1': '',
                'Tag 2': '',
                'Tag 3': '',
                'Tag 4': '',
                'Tag 5': '',
                '7 Word Problem Summary': '',
                'Emotion Triggered': '',
                'Counter-Intuitive or Counter-Narrative or Elegant Articulation': ''
            })
        # Ask follow-up questions
        prompt = "Think like the best podcast interviewer. What will be the {} best follow-up questions to ask?\n".format(num_follow_up_questions)
        for i, follow_up_question in enumerate(follow_up_questions):
            follow_up_question = st.text_input(prompt + "Question {}: ".format(i + 1), key=f"question_{i}")
            if follow_up_question.strip():
                # Search the ebook with follow-up question
                search_results = search(follow_up_question, embeddings)
                # Add search results to table data
                for result in search_results:
                    data.append({
                        'Question': follow_up_question,
                        'Paragraph/Sentence/Quote': result,
                        '30 Word Summary': '',
                        'Tag 1': '',
                        'Tag 2': '',
                        'Tag 3': '',
                                            'Tag 4': '',
                    'Tag 5': '',
                    '7 Word Problem Summary': '',
                    'Emotion Triggered': '',
                    'Counter-Intuitive or Counter-Narrative or Elegant Articulation': ''
                })

    # Display the table with the search results
    st.write(pd.DataFrame(data))
