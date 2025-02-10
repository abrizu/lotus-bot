from dotenv import load_dotenv
load_dotenv()

from keys import *

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains.retrieval import create_retrieval_chain

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools.retriever import create_retriever_tool


from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory



### Discord Deployment : Lotus ###
@client.event
async def on_ready():  # Initial deployment on discord.
    print(f'{client.user} has connected to Discord!')
    channel = client.get_channel(ALLOWED_CHANNEL_ID)

    await channel.send("Lotus is ready to assist.")

def get_retriever(): # Simple function to universally search at kwargs hardcoded value
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return retriever

def get_documents_from_web(url): # This is our reader function
    # scrape data from webpage and import the data into the chain document
    print(f"Read webpage: {url}")

    loader = WebBaseLoader(url)
    print(f"Loaded new webpage: {url}")
    docs = loader.load() # read the docs
    print("reading...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splitDocs = splitter.split_documents(docs)
    return splitDocs


### create_db and query_db functions ###
def create_db(docs): # Appends sliced url content to the database
    embedding = GoogleGenerativeAIEmbeddings(google_api_key=GENAI_TOKEN_ID, model="models/embedding-001", )
    vectorStore = PineconeVectorStore.from_documents(
        docs,
        embedding=embedding,
        index_name=pinecone_index_name
        )
    return vectorStore


def query_db_url(docs, top_k=2): # Queries the database without appending new data from url
    if isinstance(docs, list):
        docs = " ".join([doc.page_content for doc in docs])
    results = vector_store.similarity_search(docs, k=top_k, filter={"source": {"$exists": True}}) # If the link is similar to the query
    
    for res in results:
        print(f"* ({res.metadata["source"]})")

    return results

def vector_store_url(url): # This function checks if the current url is in the vector database. Appends if so, otherwise ignores it.

    docs = get_documents_from_web(url)
    results = query_db_url(docs)

    url_in_results = any(res.metadata["source"] == url for res in results)

    if not url_in_results: # if the url is not in results, add url content to the db
        vectorStore = create_db(docs)
        print(f"Created new vector store {url}")
    else:
        # uses the existing vector store db
        vectorStore = PineconeVectorStore(embedding=embeddings, index=index)
        print(f"Using existing vector store {url}")

    return vectorStore

### END create_db and query_db functions ###

### Start chaining functions ###
def create_prompt_chain(user_input):
    system_prompt = (
        """You are a helpful assistant. Answer the user's questions based on the context: {context}"""
    )

    retriever = get_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": user_input})

    return response["output"]


### Contextualize question ###
def contextualize_question():

    retriever = get_retriever()

    contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    return history_aware_retriever

### Answer question ###
def answer_question():

    history_aware_retriever = contextualize_question()

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

chat_history = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_history:
        chat_history[session_id] = ChatMessageHistory()
    return chat_history[session_id]  

def chain():
    rag_chain = answer_question()

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain

### Overloaded process_chat functions ###
def process_chat(user_input, chat_history): # Currently process chat without agents
    conversational_rag_chain = chain()

    response = conversational_rag_chain.invoke({
        "input": user_input,
        "chat_history": chat_history
        },
        config={
            "configurable": {"session_id": "ns1"}
        },
    )["answer"]

    return response

# This iteration currently does not run with agents. # It is a work in progress.
def search_web_agent():
    global agentExecutor

    search = TavilySearchResults(max_results=2)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    stock_market = stock_market_api()

    retriever_tools = create_retriever_tool({
        retriever,
        "Pinecone Retriever",
        "Use this tool when searching for information about Pinecone Queries."
    }
    
    )
    prompt = hub.pull("hwchase17/react") # temp prompt
    tools = [ search, retriever_tools ]
    agent = create_react_agent(llm, tools, prompt)

    agentExecutor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agentExecutor

### THIS IS IF IT LOADS WITH AGENT ###
def process_chat_with_agent(agentExecutor, user_input, chat_history):
    response = agentExecutor.invoke({
        "chat_history": chat_history,
        "input": user_input
    },
        config={
            "configurable": {"session_id": "agent-01"}
        },
    )["output"]

    return response

# END 

### Driver ###
if __name__ == '__main__':

    global agentExecutor

    # Default url #
    url = "https://docs.pinecone.io/guides/data/query-data#query-by-record-id"
    vector_store_url(url)

    agentExecutor = search_web_agent()

    @client.event
    async def on_message(message):

        channel_ID_listen = [ ALLOWED_CHANNEL_ID ]

        if message.author == client.user:
            return

        if message.channel.id in channel_ID_listen:
                command = message.content.strip()

                if command.lower() == 'exit':
                    await message.channel.send("Exiting...")
                    sys.exit(0)
                
                session_id = str(message.author.id)
                response = process_chat(command, session_id)
                agent_response = process_chat_with_agent(agentExecutor, command, chat_history)

                current_ai = agent_response # toggle response use

                session_history = get_session_history(session_id)
                session_history.add_message(HumanMessage(content=command))
                session_history.add_ai_message(AIMessage(content=current_ai))
                print("Assistant: ", current_ai)
                await message.channel.send(current_ai)

client.run(DISCORD_TOKEN_ID)

# AI modelling
# 
# AI agent: Functions can include adjusting personality, tonality, etc.
# Turning it into a full fledged chatbot 

# tools = [ search, retriever, adjust_sentiment, adjust_personality, adjust_tonality, play_games, etc. ]

# search: simply searches the web
# retriever: retrieves existing data from the vector database based on the query.
# adjust_sentiment: adjusts the sentiment of the response based on sentiment algorithm and labelling.
# adjust_personality: adjusts the personality of the response based on personality algorithm.
# adjust_tonality: adjusts the tonality of the response based on tonality algorithm.
# play_games: plays games with the user.

# CRITICAL ISSUES
# Need advanced chunking methods: Chunk size is dynamic depending on the content, and can overload
# the model if the chunk size is too large.

# Use semantic chunking to chunk the data into smaller pieces based on semantic meaning.