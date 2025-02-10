import getpass
import discord
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

GENAI_TOKEN_ID = os.getenv('GENAI_TOKEN')

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = GENAI_TOKEN_ID

# instantiate model 
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# prompt template 
# prompt = ChatPromptTemplate.from_template("Tell me a joke about a {obj}")
prompt = ChatPromptTemplate.from_messages(
       [
        ( "system", "Generate a list of 10 synonyms for the following word. Return the results as a comma separated list."),
        ("human", "{input}")
        ]
)

# create LLM chain
chain = prompt | llm

response = chain.invoke({"input": "garnish"})
print(response.content)