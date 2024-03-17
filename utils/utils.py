import PyPDF2
import os
import re
from typing import List, Dict, Annotated
from PyPDF2 import PdfReader
import requests
from dotenv import load_dotenv
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
load_dotenv()

HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
if HUGGING_FACE_TOKEN is None:
    raise ValueError("Missing environment variable HUGGING_FACE_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": f"Bearer {HUGGING_FACE_TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

def verify_pdf_path(file_path):
    try:
        print(file_path)
        with open(file_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            if len(pdf_reader.pages) > 0:
                pass
            else:
                raise("PDF file is empty")
    except PyPDF2.errors.PdfReadError:
        raise PyPDF2.errors.PdfReadError("Invalid PDF file")
    except FileNotFoundError:
        raise FileNotFoundError("File not found, check file address again")
    except Exception as e:
        raise(f"Error: {e}")
    

def get_text_chunks(text: str, word_limit: int) -> List[str]:
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    chunks = []
    current_chunk = []
    for sentence in sentences:
        words = sentence.split()
        if len(" ".join(current_chunk + words)) <= word_limit:
            current_chunk.extend(words)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = words
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def load_pdf(file: str, word: int) -> Dict[int, List[str]]:
    reader = PdfReader(file)
    documents = {}
    for page_no in range(len(reader.pages)):
        page = reader.pages[page_no]
        texts = page.extract_text()
        text_chunks = get_text_chunks(texts, word)
        documents[page_no] = text_chunks
    return documents