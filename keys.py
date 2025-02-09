from dotenv import load_dotenv
load_dotenv()

import os, sys, discord
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


GENAI_TOKEN_ID = os.getenv('GENAI_TOKEN')
DISCORD_TOKEN_ID = os.getenv('DISCORD_BOT_TOKEN')
ALLOWED_CHANNEL_ID = int(os.getenv('CHANNEL_TOKEN'))
PINECONE_TOKEN_ID = os.getenv('PINECONE_TOKEN')
LANGSMITH_TOKEN_ID = os.getenv('LANGSMITH_API_KEY')
TAVILY_TOKEN_ID = os.getenv('TAVILY_API_KEY')

# print(f"TAVILY_TOKEN: {TAVILY_TOKEN_ID}")

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

if not (GENAI_TOKEN_ID or DISCORD_TOKEN_ID or ALLOWED_CHANNEL_ID or PINECONE_TOKEN_ID or LANGSMITH_TOKEN_ID or TAVILY_TOKEN_ID):
    print("Missing environment variable")


# Discord Bot #
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
        max_tokens=2000,
        timeout=None,
        max_retries=1
    )