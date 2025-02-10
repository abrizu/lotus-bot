from dotenv import load_dotenv
load_dotenv()

import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.agents import AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools.retriever import create_retriever_tool

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

GENAI_TOKEN_ID = os.getenv('GENAI_TOKEN')
DISCORD_TOKEN_ID = os.getenv('DISCORD_BOT_TOKEN')
ALLOWED_CHANNEL_ID = int(os.getenv('CHANNEL_TOKEN'))
PINECONE_TOKEN_ID = os.getenv('PINECONE_TOKEN')
LANGSMITH_TOKEN_ID = os.getenv('LANGSMITH_API_KEY')
TAVILY_TOKEN_ID = os.getenv('TAVILY_API_KEY')

print(f"TAVILY_TOKEN: {TAVILY_TOKEN_ID}")

if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = GENAI_TOKEN_ID

if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = PINECONE_TOKEN_ID

if not os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_TOKEN_ID

if not os.getenv("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = LANGSMITH_TOKEN_ID

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_index_name = "lotus"

# Create Retriever
loader = WebBaseLoader("https://docs.pinecone.io/guides/data/query-data#query-by-record-id")
docs = loader.load()
    
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
splitDocs = splitter.split_documents(docs)

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_index_name = "lotus"

embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GENAI_TOKEN_ID, model="models/embedding-001", )
vectorStore = PineconeVectorStore.from_documents(
    docs,
    embedding=embeddings,
    index_name=pinecone_index_name
    )
retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.4,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a friendly assistant called Max."),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad")
# ])

prompt = hub.pull("hwchase17/react")

search = TavilySearchResults(max_results=2)
retriever_tools = create_retriever_tool(
    retriever,
    "lcel_search",
    "Use this tool when searching for information about Langchain Expression Language (LCEL)."
)
tools = [search, retriever_tools]

agent = create_react_agent(model, tools, prompt)

agentExecutor = AgentExecutor(
    agent=agent,
    tools=tools
)

def process_chat(agentExecutor, user_input, chat_history):
    response = agentExecutor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    return response["output"]

if __name__ == '__main__':
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        response = process_chat(agentExecutor, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

        print("Assistant:", response)