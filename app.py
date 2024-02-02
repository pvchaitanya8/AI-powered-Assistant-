%pip -q install langchain openai
%pip install tabulate
%pip install llama-index

# Imports
import os
import pandas as pd
import sqlite3

from langchain.agents import create_csv_agent
from langchain.llms import OpenAI

from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding

# Set the OpenAI API key in the environment variable
os.environ["OPENAI_API_KEY"] = "*********************************"
openai.api_key = os.environ["OPENAI_API_KEY"]

# Load CSV data into SQLite database
df = pd.read_csv('knowledge/data.csv')
conn = sqlite3.connect('database.db')
df.to_sql('table_name', conn, if_exists='replace')

# Function to update CSV data from SQLite database
def updateData():
    query = "SELECT * FROM table_name;"
    df = pd.read_sql_query(query, conn)
    file_path = 'knowledge/data.csv'
    df.to_csv(file_path, index=False)

# Function to create and load the TXT chatbot index
def create_and_load_txt_index(path):
    max_input = 4096
    tokens = 246
    chunk_size = 600
    max_chunk_overlap = 0.1

    promptHelper = PromptHelper(max_input, tokens, max_chunk_overlap, chunk_size_limit=chunk_size)
    llmPredictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-ada-001", max_tokens=tokens))
    docs = SimpleDirectoryReader(path).load_data()
    service_context = ServiceContext.from_defaults(llm_predictor=llmPredictor, prompt_helper=promptHelper)
    embed_model = OpenAIEmbedding(embed_batch_size=1)
    vectorIndex = GPTVectorStoreIndex.from_documents(documents=docs, service_context=service_context, embed_model=embed_model)
    vectorIndex.storage_context.persist(persist_dir='Store')

    return vectorIndex

# Function to answer questions using both CSV and TXT chatbots
def answer_question(question):
    # CSV Chatbot
    csv_agent = create_csv_agent(OpenAI(temperature=0), 'knowledge/data.csv', verbose=True)
    csv_response = csv_agent.run(question)

    # Update CSV data
    updateData()

    # TXT Chatbot
    txt_index = create_and_load_txt_index("Knowledge")
    query_engine = txt_index.as_query_engine()
    txt_response = query_engine.query(question)

    return {
        'CSV Response': csv_response,
        'TXT Response': txt_response
    }

# User input loop
while True:
    print("""
            0 - STOP
            1 - Ask Question""")
    choice = int(input())
    
    if choice == 0:
        break
    elif choice == 1:
        question = input("Enter the Question: ")
        response = answer_question(question)
        print(response)
    else:
        print("Invalid input, TRY AGAIN!")
