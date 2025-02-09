from dotenv import load_dotenv
load_dotenv()

# updated jarvisbot code

import os, discord, asyncio, json, sys
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.chains.retrieval import create_retrieval_chain

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools.retriever import create_retriever_tool


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

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GENAI_TOKEN_ID, model="models/embedding-001", )

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("lotus")
vector_store = PineconeVectorStore(embedding=embeddings, index=index)

llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.4,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )



@client.event
async def on_ready():  # Initial deployment
    print(f'{client.user} has connected to Discord!')
    channel = client.get_channel(ALLOWED_CHANNEL_ID)

    await channel.send("This is Lotus")

def get_documents_from_web(url): # This is our reader function
    # scrape data from webpage and import the data into the chain document

    loader = WebBaseLoader(url)
    print(f"Loaded new webpage: {url}")
    docs = loader.load() # read the docs
    print("reading...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20
    )
    splitDocs = splitter.split_documents(docs)
    return splitDocs

def create_db(docs): 
    # This is our writer function into the vector database
    # This function should only be called if we need to write to the database
    # Make a separate function that queries the current database
    embedding = GoogleGenerativeAIEmbeddings(google_api_key=GENAI_TOKEN_ID, model="models/embedding-001", )
    vectorStore = PineconeVectorStore.from_documents(
        docs,
        embedding=embedding,
        index_name=pinecone_index_name
        )
    return vectorStore

def query_db(docs, top_k=2): # This is our query function
    if isinstance(docs, list):
        # Assuming docs is a list of document objects, extract the text content
        docs = " ".join([doc.page_content for doc in docs])
    results = vector_store.similarity_search(docs, k=top_k, filter={"source": {"$exists": True}}) # passes the link, 
    
    for res in results:
        print(f"* ({res.metadata["source"]})")

    return results

def create_chain(vectorStore):
    model = llm

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

def search_web_agent():

    search = TavilySearchResults(max_results=2)
    # search_results = search.invoke("What is the weather in Detroit, Michigan?")
    # print(search_results)

    retriever_tools = create_retriever_tool()
    prompt = hub.pull("hwchase17/react") # temp prompt
    tools = [search, retriever_tools]
    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent_executor.invoke({"input": "How do I reduce my token usage when making LLMs?"})


if __name__ == '__main__':

    url = "https://docs.pinecone.io/guides/data/query-data#query-by-record-id"

    chain = None
    chat_history = []

    # query first, check if any top k results include link to database
    # if so, ignore creating a new db
    # if not, scrape the webpage and add to database

    search_web_agent()

    docs = get_documents_from_web(url)
    results = query_db(docs)

    url_in_results = any(res.metadata["source"] == url for res in results)

    if not url_in_results: # if the url is not in results, add url content to the db
        vectorStore = create_db(docs)
        print(f"Created new vector store {url}")
    else:
        # uses the existing vector store db
        vectorStore = PineconeVectorStore(embedding=embeddings, index=index)
        print(f"Using existing vector store {url}")

    chain = create_chain(vectorStore)

    @client.event
    async def on_message(message):

        channel_ID_listen = [ ALLOWED_CHANNEL_ID ]

        if message.author == client.user:
            return

        if message.channel.id in channel_ID_listen:
                command = message.content.strip()

                if command.lower() == 'exit':
                    await message.channel.send("EXIT")
                    return
                
                response = process_chat(chain, command, chat_history)
                chat_history.append(HumanMessage(content=command))
                chat_history.append(AIMessage(content=response))
                print("Assistant: ", response)
                await message.channel.send(response)

client.run(DISCORD_TOKEN_ID)



# issue i notice now is that chat history can be overloaded if conversation extends.
# need to figure out a way to limit the chat history


        