"""Extract text from PDF files in given path, and create .txt files"""

from PyPDF2 import PdfReader
import pandas as pd
import numpy as np
import time
import os

def substitute_symbols(original_str):
        return original_str.replace("/C0", "-").replace("/C3", "*").replace("/C14", "°")

def parse_pdfs(pdf_path, destination_path=None, substitution_function=substitute_symbols, verbose=False):
    """Extract text from PDF files in given path, and create .txt files
    
    pdf_path: path where the pdf files to be parsed are
    destination_path: path where the .txt files with the pdfs' contents will be saved. Defaults to pdf_path.
    substitution_function: function that substitutes symbols or texts in the parsed text.
    """
    if destination_path == None:
        destination_path = pdf_path

    if verbose:
        print(f"Extracting pdfs from {pdf_path} to text files in {destination_path}.")
    t=time.time()
    for filename in os.listdir(pdf_path):
        text = ''
        if filename.endswith(".pdf"):
            reader = PdfReader(os.path.join(pdf_path, filename))
            for page in reader.pages:
                text += substitution_function(page.extract_text()) + "\n"
            with open(os.path.join(destination_path, filename.replace('.pdf', '.txt')), 'w') as f:
                f.write(text) 
    if verbose:
        print('>>>> Parsed PDFs in Total Time: {}'.format(time.time()-t)) 
        print(f'Text files saved to {destination_path}')

