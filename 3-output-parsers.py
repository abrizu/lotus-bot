from dotenv import load_dotenv
load_dotenv()

import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

GENAI_TOKEN_ID = os.getenv('GENAI_TOKEN')

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = GENAI_TOKEN_ID


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

def call_string_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tell me a joke about the following subject"),
        ("human", "{input}")
    ])

    parser = StrOutputParser()
    chain = prompt | llm | parser

    return chain.invoke({
        "input": "dog"
    })

def call_list_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate a list of 10 synonyms for the following word. Return the results as a comma separated list."),
        ("human", "{input}")
    ])

    parser = CommaSeparatedListOutputParser()
    chain = prompt | llm | parser

    return chain.invoke({
        "input": "gradient"
    })

def call_json_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract information from the following phrase. \nFormatting Instructions: {format_instructions}"),
        ("human", "{phrase}")
    ])

    class Person(BaseModel):
        recipe: str = Field(description="The name of the recipe")
        ingredients: list = Field(description="Ingredients")
        age: str = Field(description="The age of the person")
        name: str = Field(description="The name of the person")

    parser = JsonOutputParser(pydantic_object=Person)

    chain = prompt | llm | parser
    return chain.invoke({
        "phrase": "The ingredients for a pepperoni pizza are tomatoes, onions, cheese, basil.",
        "format_instructions": parser.get_format_instructions()
    })


# print(call_string_output_parser())
# print(call_list_output_parser())
print(call_json_output_parser())
