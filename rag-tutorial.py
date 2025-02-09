import os, discord, bs4
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import WebBaseLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from pinecone import Pinecone

from langchain_core.documents import Document
from typing_extensions import List, TypedDict



GENAI_TOKEN_ID = os.getenv('GENAI_TOKEN')
DISCORD_TOKEN_ID = os.getenv('DISCORD_BOT_TOKEN')
ALLOWED_CHANNEL_ID = int(os.getenv('CHANNEL_TOKEN'))
PINECONE_TOKEN_ID = os.getenv('PINECONE_TOKEN')
LANGSMITH_TOKEN_ID = os.getenv('LANGSMITH_API_KEY')

if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = GENAI_TOKEN_ID

if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = PINECONE_TOKEN_ID

if not os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_TOKEN_ID

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# init llm
llm = init_chat_model("gemini-1.5-flash", model_provider="google_genai")
embedding = GoogleGenerativeAIEmbeddings(google_api_key=GENAI_TOKEN_ID, model="models/embedding-001", )

# init pinecone
pc = Pinecone(api_key=PINECONE_TOKEN_ID)
index = pc.Index("lotus")
vector_store = PineconeVectorStore(embedding=embedding, index=index)


# This section is for Indexing
# Loads blog documents into total characters passed through BeautifulSoup
# Splits document into 66 sub-documents for VectorStore

def web_loader(url):
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()

    assert len(docs) == 1
    print(f"Total characters: {len(docs[0].page_content)}")
    text_splitter(docs)

def text_splitter(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)

    print(f"Split blog post into {len(all_splits)} sub-documents.")

    document_ids = vector_store.add_documents(documents=all_splits)
    print(document_ids[:3])

# END

# Retrieval and Generation
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def rag():
    global prompt
    prompt_message = """
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
    """
    prompt = hub.pull(prompt_message)

    example_messages = prompt.invoke(
        {"content": "(context goes here)", "Question": "(question goes here)"}
    ).to_messages()

    assert len(example_messages) == 1
    print(example_messages[0].content)

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    global prompt
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


from langgraph.graph import START, StateGraph
from IPython.display import Image, display

def disp_graph():
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    display(Image(graph.get_graph().draw_mermaid_png()))

if __name__ == '__main__':
    web_loader("https://lilianweng.github.io/posts/2023-06-23-agent/")


