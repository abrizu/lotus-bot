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

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever



GENAI_TOKEN_ID = os.getenv('GENAI_TOKEN')
PINECONE_TOKEN_ID = os.getenv('PINECONE_TOKEN')
LANGSMITH_TOKEN_ID = os.getenv('LANGSMITH_API_KEY')

if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = GENAI_TOKEN_ID

if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = PINECONE_TOKEN_ID

if not os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_TOKEN_ID

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_index_name = "chatbot-scrim"

def get_documents_from_web(url):
    # scrape data from webpage and import the data into the chain document
    
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20
    )
    splitDocs = splitter.split_documents(docs)
    return splitDocs

def create_db(docs):
    embedding = GoogleGenerativeAIEmbeddings(google_api_key=GENAI_TOKEN_ID, model="models/embedding-001", )
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

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    # chain = prompt | llm
    chain = create_stuff_documents_chain(
        llm=model, 
        prompt=prompt
    )

    retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation.")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt
    )

    retrieval_chain = create_retrieval_chain(
        # retriever
        history_aware_retriever,
        chain
    )

    return retrieval_chain

def process_chat(chain, question, chat_history):
    response = chain.invoke({
        "chat_history": chat_history,
        "input": question
    })
    return response["answer"]

if __name__ == '__main__':
    docs = get_documents_from_web("https://python.langchain.com/v0.1/docs/expression_language/")
    vectorStore = create_db(docs)
    chain = create_chain(vectorStore)

    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        response = process_chat(chain, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print("Assistant: ", response)