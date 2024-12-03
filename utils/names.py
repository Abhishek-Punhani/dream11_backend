import os
from pydantic import BaseModel, Field
from typing import List
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

load_dotenv()

# Define the output structure using Pydantic
class TeamNames(BaseModel):
    names: List[str] = Field(description="List of cricket team names")

# Initialize the output parser
parser = PydanticOutputParser(pydantic_object=TeamNames)

# HuggingFace API Key
sec_key = os.getenv("HUGGINGFACE_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = sec_key

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Initialize the LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    temperature=0.8,
    model_kwargs={"max_length": 50},
    huggingfacehub_api_token=sec_key
)

# Create Prompt Template with format instructions
team_name_prompt = PromptTemplate(
    template="""
You are an AI assistant that generates creative cricket-themed team names.

{format_instructions}

Generate exactly 5 unique and catchy cricket team names that are 2-3 words long.
Each name must be only related to cricket and sound exciting and there should be no mention of any nationality.

Ensure the output is in the exact format specified above.
""",
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

def generate_team_names():
    try:
        prompt = team_name_prompt.format()
        response = llm.invoke(prompt)
        parsed_response = parser.parse(response)
        team_names = [name for name in parsed_response.names[:5]]
        return team_names
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        try:
            lines = response.strip().split('\n')
            cleaned_names = [line.strip() for line in lines 
                             if line.strip() and not line.startswith(('Team Names:', '{', '}', '"names":', '[', ']'))]
            if cleaned_names:
                return cleaned_names[:5]
        except:
            print("Failed to parse names even with fallback method.")
            return []
