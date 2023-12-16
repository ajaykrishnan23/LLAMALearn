'''
init tokenizer
init the question encoder

get the chunks as list of strings
output should be ((embedding,chunk),(embedding,chunk))

'''

from flask import Flask, request, jsonify
from transformers import RagTokenizer, DPRQuestionEncoder
import torch
import logging
import sys

log_level = logging.INFO
logging.basicConfig(stream=sys.stdout, level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')


app = Flask(__name__)

def encode_paragraphs(paragraphs):
    embeddings = []
    for paragraph in paragraphs:
        input_ids = tokenizer(paragraph, return_tensors="pt").input_ids.to(device)
        embeddings.append((question_encoder(input_ids).pooler_output.tolist()[0],paragraph))
        logging.info("Chunking %s", paragraph)
    return tuple(embeddings)

@app.route('/convert_chunks', methods=['POST'])
def get_answer():
    data = request.json
    chunks = data.get('chunks',None)
    if not chunks:
        return jsonify({'error': 'No Chunks extracted'}), 400

    embeddings = encode_paragraphs(chunks)
    
    return jsonify({'embeddings': embeddings})

if __name__ == '__main__':
    device = 'cpu'
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)
    app.run(host='0.0.0.0',debug=True)