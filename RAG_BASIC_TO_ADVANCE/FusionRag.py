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

# Replace with your MongoDB connection string
mongodb_conn_string = "mongodb+srv://Motionfog:Dropleton123@cluster0.xvptlon.mongodb.net/test"

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
    cohere_api_key="D98frCXLayLP85D3mFPmM1EbRTOMYwwArnyuPUiS",
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


# RAG-Fusion: Related
template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)


generate_queries = (
    prompt_rag_fusion 
    | ChatDeepSeek(model="deepseek-chat",
                   temperature=0,
                   max_tokens=512,
                   api_key="sk-8252b811cc0241088e75623ffd779fb0"
                   ) 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
 )

def reciprocal_rank_fusion(results:list[list], k=60):
    """Reciprocal_rank fusion that takes multiple lists of ranked documents
    and an optional parameter k used in the RRF formula"""
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list , with it's rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document a string formaat to use as a key 
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any 
            previous_score = fused_scores[doc_str]
            # Update the score of the document usng the RRF formula:1/(rank+k)
            fused_scores[doc_str] += 1 / (rank + k)
    # sort the documents based on their fused scores in descending order to get the final reranked results
    reranke_results = [
        (loads(doc), score) for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranke_results

question = "What is task decomposition for LLM agents?"
retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
docs = retrieval_chain_rag_fusion.invoke({"question" : question})



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
    api_key="sk-8252b811cc0241088e75623ffd779fb0"
)

final_rag_chain = (
    {"context": retrieval_chain_rag_fusion, 
     "question": itemgetter("question")} 
    | prompt
    | llm
    | StrOutputParser()
)

response = final_rag_chain.invoke({"question":question})
print(response, "response")