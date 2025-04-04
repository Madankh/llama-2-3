import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_deepseek import ChatDeepSeek

### INDEXING ###
loader = WebBaseLoader(
    web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_ = ("post-content", "post-title", "post_header")
        )
    ),
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# from langchain_community.embeddings import CohereEmbeddings
from langchain_cohere import CohereEmbeddings

cohere_embeddings = CohereEmbeddings(
    model="embed-english-light-v3.0",
    cohere_api_key="Key here",
     user_agent="my-app")

vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=cohere_embeddings)

retriever = vectorstore.as_retriever()

#### RETRIEVAL and GENERATION ####

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = ChatDeepSeek(model_name="deepseek-chat", temperature=0, api_key="sk-")

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
rag_chain.invoke("What are the topics under Agent System Overview")

"""Part 2: Indexing"""

question = "what kinds of pets do i like ?"
document = "My favorite pet is a cat"

import tiktoken
def num_tokens_from_string(string:str, encoding_name:str)->int:
  """Return the number of tokens is text string."""
  encoding = tiktoken.get_encoding(encoding_name)
  num_tokens = len(encoding.encode(string))
  return num_tokens

num_tokens_from_string(question, "cl100k_base")

from langchain_community.embeddings import CohereEmbeddings
cohere_embeddings = CohereEmbeddings(
    model="embed-english-light-v3.0",
    cohere_api_key="",
     user_agent="my-app")
query_result = cohere_embeddings.embed_query(question)
document_result = cohere_embeddings.embed_query(document)
len(query_result)

import numpy as np

def cosine_similarity(vec1, vec2):
  dot_product = np.dot(vec1, vec2)
  norm_vec1 = np.linalg.norm(vec1)
  norm_vec2 = np.linalg.norm(vec2)
  return dot_product / (norm_vec1 * norm_vec2)

similarity = cosine_similarity(query_result, document_result)
print("Cosine similarity : ", similarity)


from langchain_community.document_loaders import WebBaseLoader
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
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=50)

# Make splits
splits = text_splitter.split_documents(blog_docs)

# Index
from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=cohere_embeddings)

retriever = vectorstore.as_retriever()


docs = retriever.get_relevant_documents("What is Task Decomposition?")

