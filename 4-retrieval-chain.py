from dotenv import load_dotenv
load_dotenv()

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone
from langchain.chains.retrieval import create_retrieval_chain

GENAI_TOKEN_ID = os.getenv('GENAI_TOKEN')
PINECONE_TOKEN_ID = os.getenv('PINECONE_TOKEN')

if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = GENAI_TOKEN_ID

if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = PINECONE_TOKEN_ID

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_index_name = "chatbot-scrim"


def get_documents_from_web(url):
    # scrape data from webpage and import the data into the chain document
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    splitDocs = splitter.split_documents(docs)
    return splitDocs

def create_db(docs):
    embedding = GoogleGenerativeAIEmbeddings(google_api_key=GENAI_TOKEN_ID, model="models/embedding-001")
    vectorStore = Pinecone.from_documents(
        docs,
        embedding=embedding,
        index_name=pinecone_index_name
        )
    return vectorStore

def create_chain(vectorStore):
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.4,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question:
    Context: {context}
    Question: {input}
    """)

    # chain = prompt | llm
    chain = create_stuff_documents_chain(
        llm=model, 
        prompt=prompt
    )

    retriever = vectorStore.as_retriever(search_kwargs={"k": 5})

    retrieval_chain = create_retrieval_chain(
        retriever,
        chain
    )

    return retrieval_chain

docs = get_documents_from_web("https://python.langchain.com/v0.1/docs/expression_language/")
vectorStore = create_db(docs)
chain = create_chain(vectorStore)

response = chain.invoke({
    "input": "What is LCEL?"
})
print(response["context"])