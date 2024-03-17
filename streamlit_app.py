import streamlit as st
import requests, json
import PyPDF2
import re
import os
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

################################
from chromadb import Client, Settings
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader
from typing import List, Dict, Annotated
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn
from utils.utils import verify_pdf_path, get_text_chunks, load_pdf, query

ef = embedding_functions.ONNXMiniLM_L6_V2()
messages = []

# collection = CreateClient.create_collection()
client = Client(settings = Settings(persist_directory="./", is_persistent=True))
collection_ = client.get_or_create_collection(name="test", embedding_function=ef)
def clear_coll():
    if collection_.name in client.list_collections():
        # If it exists, delete it
        client.delete_collection(collection_.name)
        print("Collection deleted successfully")
    else:
       print("empty already")
 

def add_text_to_collection(file: str, word: int = 200) -> None:
    docs = load_pdf(file, word)
    docs_strings = []
    ids = []
    metadatas = []
    id = 0
    for page_no in docs.keys():      
        for doc in docs[page_no]:
            docs_strings.append(doc)
            metadatas.append({'page_no':page_no})
            ids.append(id)
            id+=1

    collection_.add(
        ids = [str(id) for id in ids],
        documents = docs_strings,
        metadatas = metadatas,
    )
    return "PDF embeddings successfully added to collection"

def query_collection(texts: str, n: int) -> List[str]:
    result = collection_.query(
                  query_texts = texts,
                  n_results = n,
                 )
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    resulting_strings = []
    for page_no, text_list in zip(metadatas, documents):
        resulting_strings.append(f"Page {page_no['page_no']}: {text_list}")
    return resulting_strings

def get_response(queried_texts: List[str],) -> List[Dict]:
    global messages
    
    messages = [
                {"role": "system", "content": "You are a helpful assistant. <s>[INST]Start the answer with <s>[INST] 'Your Answer' and end the answer with 'End of your answer'[/INST]. You will try to answer with information provided in input.[/INST]<s>[INST] Use the string  coming after 'Reference' and appears before 'ques'[/INST] <s>[INST] And will always answer the question asked in 'ques:' and \
                  answer the 'ques' using 'Reference' ellaboratively and elegantly combining information from all the pages no.[/INST]."},
                {"role": "user", "content": ''.join(queried_texts)}
                ]
    message = ' '.join([str(elem) for elem in messages])
    response = query({"inputs": message,})
    return response

def get_answer(query: str, n: int):
    queried_texts = query_collection(texts = query, n = n)
    queried_string = [''.join(text) for text in queried_texts]
    queried_string = f"Reference:{queried_string[0]}" + f"ques: {query}"
    answer = get_response(queried_texts = queried_string,)
    message = ' '.join([str(elem) for elem in answer])
    pattern = r'Your Answer: (.+?)\.'
    match = re.search(pattern, message)
    if match:
        substring_after_assistant = match.group(1)
        substring_after_assistant = substring_after_assistant.replace('\\n', '\n')
        return substring_after_assistant
    else:
        print("it is what it is")
        return message

################################


    return uploaded_pdf_path
def handle_file_upload():
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    clear_coll()
    if uploaded_file is not None:
        try:
            # Use the file object directly
            with open("artifacts/uploaded/uploaded_pdf.pdf", "wb") as f:
                f.write(uploaded_file.read())
        except Exception as e:
            st.error(f"Invalid PDF: {str(e)}")
            return None

        return "artifacts/uploaded/uploaded_pdf.pdf"

##########
def handle_query(pdf_path: str, request: str):
  query = request
  verify_pdf_path(pdf_path)
  add_text_to_collection(pdf_path)
  try:
    answer = get_answer(query, 5)  
    return {"answer": answer}
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
########
def handle_query_submission(pdf_path):
  if not pdf_path:
    return None

  query = st.text_input("Enter your question about the uploaded PDF:")
  if not query:
    return None

  #api_url = f"http://127.0.0.1:8000/query/{pdf_path}" 
  headers = {"Content-Type": "application/json"}
  data = {"query": query}

  try:
    #response = requests.post(api_url, headers=headers, json=data)
    response = handle_query(pdf_path, query)
    return response
  except requests.exceptions.RequestException as e:
    st.error(f"Error communicating with backend: {str(e)}")
    return None
def display_results(answer):
    if answer:
        st.success(f"Answer: {answer}")
    else:
        st.warning("No answer found for your query.")

if __name__ == "__main__":
    st.title("PDF Question Answering App")

    uploaded_pdf_path = handle_file_upload()
    query_answer = handle_query_submission(uploaded_pdf_path)
    display_results(query_answer)
