# load required library
import os
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.docstore.document import Document
import json



quantization_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_compute_dtype=torch.bfloat16
)

model_kwargs = {'device': 'cuda'}
embeddings = HuggingFaceEmbeddings(model_kwargs=model_kwargs)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map='auto', quantization_config=quantization_config)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=150)
llm = HuggingFacePipeline(pipeline=pipe)




from langchain_community.vectorstores import Chroma
import json
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text  # Required to use the Universal Sentence Encoder

# Load JSON data from file
json_file_path = "/content/drive/MyDrive/menu.json"
with open(json_file_path, "r") as file:
    data = json.load(file)

# Extract descriptions from JSON data
descriptions = []
for category, items in data.items():
    for item_id, item_data in items.items():
        # Check if the item_data contains a description
        if isinstance(item_data, list) and len(item_data) > 0:
            description = item_data[0]  # Assuming description is the first element
            descriptions.append(description)

# Load the Universal Sentence Encoder model
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

# Function to generate embeddings for text
def generate_embeddings(text):
    # Generate embeddings for the given text
    embeddings = embed([text])  # This will return a list of embeddings
    # Return the embeddings
    return embeddings[0]  # Return the embeddings for the first (and only) text in the list

# Generate embeddings for each description
embeddings = []
for description in descriptions:
    embedding = generate_embeddings(description)
    embeddings.append(embedding)

# Create Chroma database from the descriptions and embeddings
db = Chroma.from_documents(descriptions, embeddings, persist_directory="test_index")
db.persist()


# Load the database
vectordb = Chroma(persist_directory="test_index", embedding_function = embeddings)

# Load the retriver
retriever = vectordb.as_retriever(search_kwargs = {"k" : 3})
qna_prompt_template="""###  If the data doesn't contain the answer to the question, then you must return 'Not enough information.'

{context}

### Question: {question} """

PROMPT = PromptTemplate(
   template=qna_prompt_template, input_variables=["context", "question"]
)
chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)



# A utility function for answer generation
def ask(question):
   context = retriever.get_relevant_documents(question)
   print(context)

   answer = (chain({"input_documents": context, "question": question}, return_only_outputs=True))['output_text']
   return answer




# Take the user input and call the function to generate output
user_question = input("User: ")
answer = ask(user_question)
print("Answer:", answer)