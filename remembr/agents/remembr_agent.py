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


class Position(BaseModel):
    x: float = Field(description="x coordiante")
    y: float = Field(description="y coordiante")
    z: float = Field(description="z coordiante")


class AgentAnswer(BaseModel):
    type: Literal["position", "binary", "time", "text"] = Field(
        description="input the type of answer that is expected based only on the question: position, binary, time, or text. Be sure to then fill in that selected category"
    )
    text: str = Field(
        description="Give a short reasoning for your answer."
    )
    binary: bool = Field(description="a yes/no answer for binary questions")
    position: Position = Field(description="An answer for spatial question with a position containing x,y,z coordinates.")
    orientation: float = Field(description="orientation in yaw")
    duration: float = Field(description="An answer for temporal questions that a ask for a duration. Duration is in minutes")
    time: float = Field(description="An answer for temporal questions that ask for certain timepoint in the past. Time is in minutes past since the current time given in context.")


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
            response_format=AgentAnswer,
        )

    def create_tools(self, memory: Memory):
        @tool
        def retrieve_from_text(query: str):
            """Search and return information from your video memory in the form of captions

            Args:
                x: The query that will be searched by the vector similarity-based retriever. Text embeddings of this description are used. There should always be text in here as a response! Based on the question and your context, decide what text to search for in the database. This query argument should be a phrase such as 'a crowd gathering' or 'a green car driving down the road'. The query will then search your memories for you.
            """
            print("calling tool text")
            return memory.search_by_text(query)

        @tool
        def retrieve_from_position(x: float, y: float, z: float):
            """The query that will be searched by finding the nearest memories at this x,y,z position. Based on the question and your context, decide what position to search for in the database. This query argument should be a position such as x=0.5 y=0.2 z=0.1. The query will then search your memories for you.
            Args:
                x: x coordiante
                y: y coordiante
                z: z coordiante
            """
            print("calling tool position")
            return memory.search_by_position((x, y, z))

        @tool
        def retrieve_from_time(time: str):
            """Search and return information from your video memory by using time in a format of H:M:S, like 08:32:12
            Args:
                x: The query that will be searched by finding the nearest memories at a specific time in H:M:S format. The query must be a string containing only time. Based on the question and your context, decide what time to search for in the database. This query argument should be an HMS time such as 08:02:03 with leading zeros. The query will then search your memories for you.
            """
            print("calling tool time")
            return memory.search_by_time(time)

        self.tool_list = [
            retrieve_from_text,
            retrieve_from_time,
            retrieve_from_position,
        ]

    ### Nodes

    def query(self, query: str):
        print(query)
        out = self.agent.invoke({"messages": [{"role": "user", "content": query}]})
        print(out)
        res: AgentAnswer = out["structured_response"]

        response = AgentOutput(
            type=res.type,
            text=res.type,
            binary="yes" if res.binary else "no",
            position=(res.position.x, res.position.y, res.position.x),
            orientation=res.orientation,
            duration=res.duration,
            time=res.time,
        )

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
