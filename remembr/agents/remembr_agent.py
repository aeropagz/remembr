from typing import Literal
import sys
from langchain.agents import create_agent

# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


from langchain.tools import tool


import os

sys.path.append(sys.path[0] + "/..")


from utils.util import file_to_string
from memory.memory import Memory

from pydantic import BaseModel, Field
from agents.agent import Agent, AgentOutput


class AgentAnswer(BaseModel):
    type: Literal["position", "binary", "time", "text"] = Field(
        description="input the type of answer that is expected based only on the question: position, binary, time, or text. Be sure to then fill in that selected category"
    )
    text: str = Field(
        description="a text answer here. This should be as if you are responding to a user, so do not provide low-level details."
    )
    binary: bool = Field(description="a yes/no answer")
    position: tuple[float, float, float] = Field(
        description="Position in [x,y,z] coordinates."
    )
    orientation: float = Field(description="orientation in yaw")
    duration: float = Field(description="Duration in minutes")
    time: float = Field(description="Time in minutes ago")


class ReMEmbRAgent(Agent):
    def __init__(self, temperature=0):
        # Wrapper that handles everything
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview", temperature=temperature
        )

        self.temperature = temperature

        self.chat = llm
        ### Load vectorstore
        self.embeddings = HuggingFaceEmbeddings(
            model_name="mixedbread-ai/mxbai-embed-large-v1"
        )

        # self.update_for_instance() # ref_time is None this time
        top_level_path = str(os.path.dirname(__file__)) + "/../"
        self.agent_prompt = file_to_string(
            top_level_path + "prompts/agent_system_prompt.txt"
        )


    def set_memory(self, memory: Memory):
        self.memory = memory
        self.create_tools(memory)
        self.agent = create_agent(
            model=self.chat,
            tools=self.tool_list,
            system_prompt=self.agent_prompt,
        )

    def create_tools(self, memory):
        @tool
        def retrieve_from_text(x: str):
            """Search and return information from your video memory in the form of captions

            Args:
                x: The query that will be searched by the vector similarity-based retriever. Text embeddings of this description are used. There should always be text in here as a response! Based on the question and your context, decide what text to search for in the database. This query argument should be a phrase such as 'a crowd gathering' or 'a green car driving down the road'. The query will then search your memories for you.
            """
            return memory.search_by_text(x)

        @tool
        def retrieve_from_position(x: tuple):
            """Search and return information from your video memory by using a position array such as (x,y,z)

            Args:
                x: The query that will be searched by finding the nearest memories at this (x,y,z) position. The query must be an (x,y,z) array with floating point values Based on the question and your context, decide what position to search for in the database. This query argument should be a position such as (0.5, 0.2, 0.1). They should NOT be a string. The query will then search your memories for you.
            """
            return memory.search_by_position(x)

        @tool
        def retrieve_from_time(x: str):
            """Search and return information from your video memory by using time in a format of H:M:S, like 08:32:12
            Args:
                x: The query that will be searched by finding the nearest memories at a specific time in H:M:S format. The query must be a string containing only time. Based on the question and your context, decide what time to search for in the database. This query argument should be an HMS time such as 08:02:03 with leading zeros. The query will then search your memories for you.
            """
            return memory.search_by_position(x)

        self.tool_list = [
            retrieve_from_text,
            retrieve_from_position,
            retrieve_from_time,
        ]

    ### Nodes

    def query(self, query: str):
        res = self.agent.invoke({"messages": [{"role": "user", "content": query}]})


        response = AgentOutput.from_dict(res['structured_response'].model_dump())

        return response


if __name__ == "__main__":
    from memory.milvus_memory import MilvusMemory

    # llm_name =
    # Options: 'nim/meta/llama-3.1-405b-instruct', 'gpt-4o', or any Ollama LLMs (such as 'codestral')
    memory = MilvusMemory("test", db_ip="127.0.0.1")

    agent = ReMEmbRAgent()

    agent.set_memory(memory)

    response = agent.query("Where can I sit?")
    response = agent.query_position("Where can I sit?")
