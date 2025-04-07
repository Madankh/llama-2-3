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

### Router
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# set embeddings
load_dotenv()
Cohere = os.getenv("COHERE_API_KEY")
llmKey =  os.getenv("llm")
mongodb_conn_string = os.getenv("mongodb")

# Check if keys are loaded
if not all([Cohere, llmKey, mongodb_conn_string]):
    raise ValueError("One or more environment variables (COHERE_API_KEY, llm, mongodb) are missing.")

# embd = CohereEmbeddings() # You defined cohere_embeddings later, this line is redundant

# Create a MongoDB client
try:
    client = MongoClient(mongodb_conn_string)
    client.admin.command('ping') # Verify connection
    print("MongoDB connection successful.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit() # Exit if DB connection fails

# Docs to index
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load
print("Loading documents...")
try:
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    print(f"Loaded {len(docs_list)} documents.")
except Exception as e:
    print(f"Error loading documents: {e}")
    exit()

# split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)
print(f"Split documents into {len(doc_splits)} chunks.")
# print(doc_splits, "splits")
# Index
print("Initializing embeddings...")
try:
    cohere_embeddings = CohereEmbeddings(
        model="embed-english-light-v3.0",
        cohere_api_key=Cohere,
         user_agent="langchain-router-debug" # Good practice to set user agent
    )
except Exception as e:
    print(f"Error initializing Cohere embeddings: {e}")
    exit()

db_name = "DBbase"
collection_name = "memory_router_test" # Use a distinct collection name
collection = client[db_name][collection_name]

# --- Optional: Clear collection before indexing if testing ---
# print(f"Clearing collection {db_name}.{collection_name}...")
# collection.delete_many({})
# -------------------------------------------------------------

# Create the vector store using MongoDB Atlas
print("Indexing documents into MongoDB Atlas Vector Search...")
try:
    vectorstore = MongoDBAtlasVectorSearch.from_documents(
        documents=doc_splits,
        embedding=cohere_embeddings,
        collection=collection,
        index_name="default"  # <<< MAKE SURE THIS INDEX NAME EXISTS IN ATLAS
    )
    print("Indexing complete.")
except Exception as e:
    print(f"Error creating/populating vector store: {e}")
    print("Please ensure the MongoDB connection is valid and the index 'vector_index' exists in Atlas for the collection.")
    exit()

retriever = vectorstore.as_retriever()

# Data Model
class web_search(BaseModel):
    """Select this route to perform a general web search.""" # Add docstring for clarity
    query:str = Field(..., description="The query to use when searching the internet.")

class vectorstore_search(BaseModel): # Renamed for clarity
    """Select this route to search the vectorstore containing information on agents, prompt engineering, and adversarial attacks.""" # Add docstring
    query:str = Field(..., description="The query to use when searching the vectorstore.")

# System prompt containing routing instructions
system_prompt = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to AI agents, prompt engineering, and adversarial attacks against LLMs.
Use the vectorstore_search for questions specifically about these topics based on the provided documents.
Otherwise, use web_search for general questions, current events, or topics outside the scope of the documents."""

# LLM with tool use and preamble
# Use temperature 0 for deterministic routing
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0, # Set temp to 0 for more deterministic routing
    max_tokens=512,
    api_key=llmKey)

# Bind the tools (Pydantic models) to the LLM
structured_llm_router = llm.bind_tools([web_search, vectorstore_search])

# Create the prompt template including the system message
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt), # <-- Include the routing instructions here
        ("human", "{question}"),
    ]
)

# Create the actual router chain by combining the prompt and the LLM (with tools)
question_router = route_prompt | structured_llm_router

# --- Now invoke the CHAIN ---

print("\n--- Routing Test 1 (Vectorstore expected) ---")
question1 = "what are the types of agents memory?"
print(f"Question: {question1}")
try:
    response1 = question_router.invoke({"question": question1})
    # Access tool calls directly from the response object
    print(f"Tool Calls: {response1.tool_calls}")
    # You can check the type of tool called if needed
    if response1.tool_calls:
        print(f"Tool Called: {response1.tool_calls}")
        print(f"Tool Arguments: {response1.tool_calls[0]['args']}")
except Exception as e:
    print(f"Error during invocation 1: {e}")

print("\n--- Routing Test 2 (Web Search expected) ---")
question2 = "What's the weather like in London today?"
print(f"Question: {question2}")
try:
    response2 = question_router.invoke({"question": question2})
    print(f"Tool Calls: {response2.tool_calls}")
    if response2.tool_calls:
        print(f"Tool Called: {response2.tool_calls[0]['name']}")
        print(f"Tool Arguments: {response2.tool_calls[0]['args']}")
except Exception as e:
    print(f"Error during invocation 2: {e}")

print("\n--- Routing Test 3 (No tool expected / Direct Answer - depends on LLM) ---")

question3 = "Hi how are you?"
print(f"Question: {question3}")
try:
    response3 = question_router.invoke({"question": question3})
    print(f"Tool Calls: {response3.tool_calls}") # Often [] or None for simple chat
    if response3.tool_calls:
         print(f"Tool Called: {response3.tool_calls[0]['name']}")
         print(f"Tool Arguments: {response3.tool_calls[0]['args']}")
    else:
        # Print the direct LLM response content if no tool was called
        print(f"Direct Response: {response3.content}")
except Exception as e:
    print(f"Error during invocation 3: {e}")

# Regarding your original check:
# response.tool_calls is the standard way to access tool calls in recent LangChain versions.
# response.response_metadata might contain tool calls in older versions or specific setups,
# but response.tool_calls is preferred.
# If response.tool_calls is empty ([]), then no tool was called.

# Example check:
print(f"\nDid Test 3 call a tool? {'Yes' if response3.tool_calls else 'No'}")

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# Grader Instructions (formerly preamble)
grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

# LLM for the grader
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0, # Use temperature 0 for deterministic grading
    api_key=llmKey
)

# LLM with structured output, NO preamble argument here
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt for the grader, INCLUDING instructions as a system message
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", grader_instructions), # <-- Include instructions here
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# Grader chain
retrieval_grader = grade_prompt | structured_llm_grader

# --- Test the Grader ---
question = "types of agent memory"
print(f"Retrieving documents for: {question}")
retriever_chain = retriever.map() | StrOutputParser() 
try:
    docs = retriever_chain.invoke(question)
    if not docs:
        print("Retriever returned no documents.")
    else:
        print(f"Retrieved {len(docs)} documents. Grading the second document.")
        # Ensure there is a second document before accessing it
        if len(docs) > 1:
            doc_text = docs[1].page_content # Use doc_text instead of doc_tex
            print(f"\n--- Grading Document ---\n{doc_text[:500]}...\n----------------------")
            response = retrieval_grader.invoke({"question": question, "document": doc_text})
            print("\nGrader Response:")
            print(response)
        else:
            print("Not enough documents retrieved to grade the second one.")

except Exception as e:
    print(f"An error occurred during retrieval or grading: {e}")


# --- Generation Setup ---

# Generation Instructions (preamble for the generator)
generator_instructions = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."""

llm_generator = ChatDeepSeek(model="deepseek-chat", temperature=0, api_key=llmKey)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", generator_instructions + "\n\nContext:\n{context}"), # Include instructions and context placeholder
        ("human", "Question: {question}"), # Keep human message clean
    ]
)

# Generation Chain
rag_chain = (
    # Pass question and documents through, format documents for the prompt
    {"context": lambda x: format_docs(x["documents"]), "question": itemgetter("question")}
    | generation_prompt
    | llm_generator
    | StrOutputParser()
)
if docs: 
    print("\n--- Generating Answer ---")
    try:
        relevant_docs = docs # Replace this with filtered docs if grader logic is added
        generation = rag_chain.invoke({"documents": relevant_docs, "question": question})
        print("\nGenerated Answer:")
        print(generation)
    except Exception as e:
        print(f"An error occurred during generation: {e}")
else:
    print("\nSkipping generation because no documents were retrieved.")