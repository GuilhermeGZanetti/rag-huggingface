import src.pdf_parser as pdf_parser
from src.index import create_dataset, load_dataset
from src.rag import load_rag, generate_answer
import numpy as np

#First we take the source pdf files and transform them to .txt files
#pdf_parser.parse_pdfs(pdf_path='./texts', destination_path='./texts', verbose=True)
#pdf_parser.parse_pdfs(pdf_path='./texts-test', destination_path='./texts-test', verbose=True)

#We then read the .txt files, divide them into chunks and embed them and store in faiss index
dataset = create_dataset(txt_path = './texts', verbose=True)

dataset = load_dataset()
print(dataset)


#rag_sys = load_rag(dataset, verbose=True)
#print("\n\nAnswer: ", generate_answer(rag_sys, question="What is reiforcement learning?"))

