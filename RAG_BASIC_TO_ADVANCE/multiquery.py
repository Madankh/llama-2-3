import bs4
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain.load import dumps, loads
from dotenv import load_dotenv
import os
load_dotenv()
Cohere = os.getenv("COHERE_API_KEY")
llmKey =  os.getenv("llm")
mongodb_conn_string = os.getenv("mongodb")

# Create a MongoDB client
client = MongoClient(mongodb_conn_string)

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
blog_docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50
    )

splits = text_splitter.split_documents(blog_docs)

# Index
cohere_embeddings = CohereEmbeddings(
    model="embed-english-light-v3.0",
    cohere_api_key=Cohere,
     user_agent="my-app")


db_name = "DBbase"
collection_name = "memory"
collection = client[db_name][collection_name]
client = MongoClient(mongodb_conn_string)

# Create the vector store using MongoDB Atlas
vectorstore = MongoDBAtlasVectorSearch.from_documents(
    documents=splits,
    embedding=cohere_embeddings,
    collection=collection,
    index_name="default"  # The name of your vector search index
)


retriever = vectorstore.as_retriever()


# Multi Query: Different Perspectives
template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_perspectives 
    | ChatDeepSeek(model="deepseek-chat",
                   temperature=0,
                   max_tokens=512,
                   api_key=llmKey
                   ) 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

# Retrieve
question = "What is task decomposition for LLM agents?"
retrieval_chain = generate_queries | retriever.map() | get_unique_union
docs = retrieval_chain.invoke({"question":question})


# RAG
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=512,
    api_key=llmKey
)

final_rag_chain = (
    {"context": retrieval_chain, 
     "question": itemgetter("question")} 
    | prompt
    | llm
    | StrOutputParser()
)

response = final_rag_chain.invoke({"question":question})
print(response, "response")