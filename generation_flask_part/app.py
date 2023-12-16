from flask import Flask, request, jsonify
import sys
import torch
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration, DPRQuestionEncoder
from llama_cpp import Llama
import logging
import os
from functools import wraps

# Fetch the API key from an environment variable
API_KEY = os.environ.get('API_KEY')

log_level = logging.INFO
logging.basicConfig(stream=sys.stdout, level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')


app = Flask(__name__)


# llm = Llama(
#   model_path="./neuralhermes-2.5-mistral-7b.Q2_K.gguf",  # Download the model file first
#   n_ctx=32768,  # The max sequence length to use - note that longer sequence lengths require much more resources
#   n_threads=8            # The number of CPU threads to use, tailor to your system and the resulting performance
# #   n_gpu_layers=35         # The number of layers to offload to GPU, if you have GPU acceleration available
# )

# Simple inference example
# output = llm(
#   "<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant", # Prompt
#   max_tokens=512,  # Generate up to 512 tokens
#   stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
#   echo=False        # Whether to echo the prompt
# )


# dataset = [
#     "BigQuery is a powerful cloud-based data warehousing and analytics platform developed by Google. It is designed to handle vast amounts of data and perform complex queries in a scalable and efficient manner. BigQuery operates on a serverless architecture, eliminating the need for users to manage infrastructure, provisioning, or tuning, making it an attractive choice for organizations seeking to harness the potential of big data without the operational overhead.",
#     "One of BigQuery's key strengths lies in its ability to process structured and semi-structured data stored in various formats, such as JSON, Parquet, or CSV, making it versatile for different use cases. Its integration with Google Cloud Storage allows users to easily import and export data, facilitating data sharing and collaboration. Moreover, BigQuery's support for standard SQL makes it accessible to users with SQL proficiency, while also offering advanced features like window functions and geographic functions for complex analytical tasks.",
#     "BigQuery's query execution is distributed and parallelized across a vast network of resources, enabling high-speed query performance even on petabyte-scale datasets. It offers a variety of querying options, including interactive ad-hoc queries, batch processing, and scheduled queries, ensuring flexibility for different business requirements. Additionally, BigQuery's integration with machine learning tools like TensorFlow enables data scientists to build and deploy models directly within the platform.",
#     "The security and governance features of BigQuery are robust, with fine-grained access control, audit logging, and data encryption at rest and in transit. Organizations can also leverage BigQuery's integration with Identity and Access Management (IAM) to manage permissions effectively. Furthermore, BigQuery offers a range of cost management tools, such as query optimization and slot reservation, to control expenses and optimize query performance.",
#     "In conclusion, BigQuery is a comprehensive data analytics solution that empowers organizations to extract valuable insights from massive datasets with ease. Its scalability, performance, versatility, and integration with other Google Cloud services position it as a leading choice for businesses seeking to make data-driven decisions in today's data-centric world."    
# ]

# Function to encode paragraphs and return embeddings
# def encode_paragraphs(paragraphs):
#     embeddings = []
#     for paragraph in paragraphs:
#         input_ids = tokenizer(paragraph, return_tensors="pt").input_ids.to(device)
#         embeddings.append(question_encoder(input_ids).pooler_output)
#     return embeddings



# def get_top_paragraph_indices(question, n_docs=3):
#     # Encode the question
#     input_ids = tokenizer(question, return_tensors="pt").input_ids.to(device)
#     question_embedding = question_encoder(input_ids).pooler_output

#     # Compute similarities and retrieve top paragraphs
#     doc_scores = torch.tensor([torch.cosine_similarity(question_embedding, p_emb, dim=1) for p_emb in encoded_paragraphs])
#     top_doc_indices = doc_scores.argsort(descending=True)[:n_docs]  # Retrieve top n_docs paragraphs
#     context = ''
#     for idx in top_doc_indices.tolist():
#         context += dataset[idx] + "\n"
#     return context

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('Api-Key')
        if not api_key or api_key != API_KEY:
            return jsonify({'error': 'Unauthorized'}), 401

        return f(*args, **kwargs)
    return decorated_function



# 540009924757.dkr.ecr.us-east-1.amazonaws.com/rag-generator
def generate_answer(question,context):
    system_message = f"instructions: Answer the question only if it is directly related to the context provided. If the question is beyond the scope of the context, respond with 'The question is beyond the context scope and I do not have enough information to answer.'\n \
    The context for the question is: {context} \n"
    output = llm(
    f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant", # Prompt
    max_tokens=512,  # Generate up to 512 tokens
    stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
    echo=False        # Whether to echo the prompt
    )
    return output

# def generate_answer(question,context):
#     output = llm.create_chat_completion(
#     messages = [
#         {"role": "system", "content": f"You are to answer the question given the prompt. If you do not know the answer just say I do not know. \
#          Do not be playful with your answers. This is for an application and we require serious answers. Just be direct and concise. \
#          And use the text that comes after the colon as prompt to answer any question the user asks: {context}"},
#         {
#             "role": "user",
#             "content": f"{question}"
#         }
#     ]
#     )
#     return output

@app.route('/get_answer', methods=['POST'])
@require_api_key
def get_answer():
    data = request.json
    question = data.get('question')
    context = data.get('context')
    if not question or not context:
        return jsonify({'error': 'No question provided'}), 400

    # context = get_top_paragraph_indices(question)
    logging.info("Top paragraph indices: %s", context)
    logging.info("QUESTION: %s", question)
    answer = generate_answer(question, context)
    
    return jsonify({'question':question,'answer': answer})


if __name__ == '__main__':
    llm = Llama(
    model_path="./neuralhermes-2.5-mistral-7b.Q2_K.gguf",  # Download the model file first
    n_threads=4,            # The number of CPU threads to use, tailor to your system and the resulting performance
    )
    # llm = Llama(model_path="./neuralhermes-2.5-mistral-7b.Q2_K.gguf", chat_format="llama-2")  # Set chat_format according to the model you are using
    device = 'cpu'
    # tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    # question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)
    # encoded_paragraphs = encode_paragraphs(dataset)
    app.run(host='0.0.0.0',debug=True)