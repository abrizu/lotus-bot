# This mini folder includes langchain utilization before implementation into jarvisbot

import getpass
import discord
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
GENAI_TOKEN_ID = os.getenv('GENAI_TOKEN')
DISCORD_TOKEN_ID = os.getenv('DISCORD_BOT_TOKEN')
ALLOWED_CHANNEL_ID = int(os.getenv('CHANNEL_TOKEN'))
PINECONE_TOKEN_ID = os.getenv('PINECONE_TOKEN')

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = GENAI_TOKEN_ID

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)


messages = [
    (
        "system",
        "You are a chatbot named Mimi, a cute girl who is slightly affectionate towards the user. Converse with the user.",
    ),
        ("human", "Write a poem about me."),
]

response = llm.stream(messages)
# print(response.content)

for chunk in response:
    print(chunk.content, end="", flush=True)