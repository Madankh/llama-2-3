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
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from typing import Literal,List
from langchain_core.prompts import ChatPromptTemplate
from pydantic.v1 import BaseModel, Field
from langchain_deepseek import ChatDeepSeek
from langchain_core.runnables import RunnableMap, RunnablePassthrough
import os

### Router
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# set embeddings
load_dotenv()
Cohere = os.getenv("COHERE_API_KEY")
llmKey =  os.getenv("llm")
mongodb_conn_string = os.getenv("mongodb")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

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
collection_name = "memory_router" # Use a distinct collection name
collection = client[db_name][collection_name]

print("Indexing documents into MongoDB Atlas Vector Search...")
try:
    vectorstore = MongoDBAtlasVectorSearch.from_documents(
        documents=doc_splits,
        embedding=cohere_embeddings,
        collection=collection,
        index_name="vector_index",
        relevance_score_fn="cosine",
    )
    print("Indexing complete.")
except Exception as e:
    print(f"Error creating/populating vector store: {e}")
    print("Please ensure the MongoDB connection is valid and the index 'vector_index' exists in Atlas for the collection.")
    exit()

retriever = vectorstore.as_retriever()

# Data model 
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search", "llm_fallback"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )
llm = ChatDeepSeek(model="deepseek-chat", temperature=0,api_key=llmKey)
structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt
system = """You are an expert at routing a user question to a vectorstore, web search, or llm_fallback.
            The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
            Use the vectorstore for questions on these topics.
            For scientific questions requiring established knowledge, use llm_fallback.
            For questions requiring current information or topics outside your knowledge base, use web-search."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
print(
    question_router.invoke(
        {"question": "Who will the Bears draft first in the NFL draft?"}
    )
)
print(question_router.invoke({"question": "What are the types of agent memory?"}))

#### Retrieval Grader
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

llm = ChatDeepSeek(model="deepseek-chat", temperature=0, api_key=llmKey)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""


grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader  = grade_prompt | structured_llm_grader

# Retrieve documents
question = "types of agent memory"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content  # Get the second document's content
response = retrieval_grader.invoke({
    "question": question, 
    "document": doc_txt
})
print(response)


### Generate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Document: {document}
Question: {question}
Answer:""")

#LLM
llm = ChatDeepSeek(model="deepseek-chat", temperature=0, api_key=llmKey)

# post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# chain
rag_chain = prompt | llm | StrOutputParser()

# Format the documents and run the chain
formatted_docs = format_docs(docs)
generation = rag_chain.invoke({"document": formatted_docs, "question": question})
print(generation)

### LLM fallback
### LLM fallback

from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
# Preamble
preamble = """You are an assistant for question-answering tasks. Answer the question based upon your knowledge. Use three sentences maximum and keep the answer concise."""

# LLM
llm = ChatDeepSeek(model="deepseek-chat", temperature=0, api_key=llmKey)


# Prompt
def prompt(x):
    return ChatPromptTemplate.from_messages(
        [HumanMessage(f"Question: {x['question']} \nAnswer: ")]
    )


# Chain
llm_chain = prompt | llm | StrOutputParser()

# Run
question = "Hi how are you?"
generation = llm_chain.invoke({"question": question})
print(generation)


### Hallucination Grader
class GradeHallucinations(BaseModel):
    """
    Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# LLM with function call
llm = ChatDeepSeek(model="deepseek-chat", temperature=0, api_key=llmKey)
structured_llm_grader = llm.with_structured_output(
    GradeHallucinations
)

# Prompt
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        # ("system", preamble),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader
hallucination = hallucination_grader.invoke({"documents": docs, "generation": generation})
print(hallucination, "hallucination")


# Data model
class GradeAnswer(BaseModel):
    """
    Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.
    Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

# LLM with function call
llm = ChatDeepSeek(model="deepseek-chat", temperature=0, api_key=llmKey)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# Prompt
answer_prompt = ChatPromptTemplate.from_messages(
    
    ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),

)

answer_grader = answer_prompt | structured_llm_grader
answer =answer_grader.invoke({"question": question, "generation": generation})
print(answer, "answer_grader")

####  Search
from langchain_community.tools.tavily_search import TavilySearchResults
web_search_tool = TavilySearchResults()

from typing_extensions import TypedDict

class GraphState(TypedDict):
    """
    Represents the state of our graph
    """
    question:str
    generation:str
    documents:List[str]

from langchain.schema import Document

def retrieve(state):
    """
    Retrive docs

    """
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents":documents, "question":question}

def llm_fallback(state):
    question = state["question"]
    generation = llm_chain.invoke({"question":question})
    return {"question":question, "generation":generation}

def generate(state):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    if not isinstance(documents, list):
        documents = [documents]

    # Format documents to a single string
    formatted_docs = format_docs(documents)  
    
    # Pass document (singular) to match the prompt template
    generation = rag_chain.invoke({"document": formatted_docs, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}


def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    return {"documents": web_results, "question": question}


### Edges ###


def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})

    # Choose datasource
    datasource = source.datasource
    if datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    else:
        print("---ROUTE QUESTION TO LLM---")
        return "llm_fallback"


def decide_to_generate(state):
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, WEB SEARCH---")
        return "web_search"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"




import pprint
from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("web_search", web_search)
workflow.add_node("retrieve", retriever)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node('generate', generate)
workflow.add_node('llm_fallback', llm_fallback)

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
        "llm_fallback": "llm_fallback",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "web_search": "web_search",
        "generate": "generate",
    },
)
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",  # Hallucinations: re-generate
        "not useful": "web_search",  # Fails to answer question: fall-back to web-search
        "useful": END,
    },
)
workflow.add_edge("llm_fallback", END)

# Compile
app = workflow.compile()

# Run
inputs = {
    "question": "what is atom?"
}

for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint.pprint(f"Node '{key}':")
        # Optional: print full state at each node
    pprint.pprint("\n---\n")

# Final generation
pprint.pprint(value["generation"])