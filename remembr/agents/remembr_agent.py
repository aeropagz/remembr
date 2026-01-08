from typing import Annotated, Sequence, TypedDict, List, Optional
import os
import re
import traceback
import sys

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool
from langgraph.prebuilt import create_react_agent

import sys
sys.path.append(sys.path[0] + '/..')

from remembr.utils.util import file_to_string
from remembr.agents.agent import Agent, AgentOutput
from remembr.memory.memory import Memory

class AgentResponseSchema(BaseModel):
    type_reasoning: str = Field(description="Reasoning for why this type of answer was selected based on the question.")
    type: str = Field(description="The category of the answer: 'position', 'binary', 'time', or 'text'.")
    answer_reasoning: str = Field(description="Reasoning for the specific answer provided, explaining how the context supports it.")
    text: str = Field(description="A natural language response to the user's question.")
    binary: Optional[str] = Field(description="For binary questions, 'yes' or 'no'. Otherwise null.", default=None)
    position: Optional[List[float]] = Field(description="A list of 3 floats [x, y, z] if a location is relevant. Otherwise null.", default=None)
    orientation: Optional[float] = Field(description="A float [theta] representing the orientation if position is provided. Otherwise null.", default=None)
    time: Optional[float] = Field(description="The time in minutes ago (as a float) if the question is 'when'. Otherwise null.", default=None)
    duration: Optional[float] = Field(description="The duration in minutes (as a float) if the question is 'how long'. Otherwise null.", default=None)

class ReMEmbRAgent(Agent):
    def __init__(self, llm_type='gpt', temperature=0):
        self.llm_type = llm_type
        self.temperature = temperature
        self.chat = self.llm_selector(llm_type, temperature)
        self.embeddings = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')

        top_level_path = os.path.join(os.path.dirname(__file__), '..')
        self.base_agent_prompt = file_to_string(os.path.join(top_level_path, 'prompts/agent_system_prompt.txt'))
        self.base_generate_prompt = file_to_string(os.path.join(top_level_path, 'prompts/generate_system_prompt.txt'))

        self.memory = None
        self.graph = None

    def llm_selector(self, llm_type, temperature):
        if 'gpt' in llm_type:
            model_name = "gpt-5.1" if llm_type == 'gpt' else llm_type
            return ChatOpenAI(model=model_name, temperature=temperature)
        elif 'gemini' in llm_type:
            model_name = "gemini-2.5-flash" if llm_type == 'gemini' else llm_type
            return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        raise Exception(f"No correct LLM provided: {llm_type}")

    def set_memory(self, memory: Memory):
        self.memory = memory
        tools = self.create_tools(memory)
        
        system_instructions = (
            "You are a robot assistant with access to memories of what you have seen. "
            "Use the provided tools to search through your memories to answer the user's question. "
            "Reason step-by-step about what info you need. "
            "XYZ coordinates are in meters. Time is in HH:MM:SS format in tools, but answers should be in minutes ago. "
            "When you have enough information, provide a final comprehensive answer."
        )
        
        self.graph = create_react_agent(self.chat, tools, state_modifier=system_instructions)

    def create_tools(self, memory):
        def retrieve_text(query: str) -> str:
            """Search and return information from video memory in the form of captions. 
            The query should be a text description of what you are looking for."""
            return memory.search_by_text(query)

        def retrieve_position(x: float, y: float, z: float) -> str:
            """Search and return information from video memory by using a (x,y,z) position."""
            return memory.search_by_position((x, y, z))

        def retrieve_time(hms_time: str) -> str:
            """Search and return information from video memory by using an H:M:S time (e.g., '08:02:03')."""
            return memory.search_by_time(hms_time)

        return [
            StructuredTool.from_function(func=retrieve_text, name="retrieve_from_text", description="Search by text description"),
            StructuredTool.from_function(func=retrieve_position, name="retrieve_from_position", description="Search by (x,y,z) coordinates"),
            StructuredTool.from_function(func=retrieve_time, name="retrieve_from_time", description="Search by H:M:S time")
        ]

    def query(self, question: str) -> AgentOutput:
        if not self.graph:
            raise Exception("Memory not set. Call set_memory() first.")

        inputs = {"messages": [HumanMessage(content=question)]}
        try:
            result = self.graph.invoke(inputs)
        except Exception as e:
            print(f"Error during agent execution: {e}")
            traceback.print_exc()
            raise e

        formatter = self.chat.with_structured_output(AgentResponseSchema)
        history = result["messages"]
        
        formatting_instruction = (
            "Based on the conversation above, provide a structured response. "
            "Follow these rules:\n"
            "1. category must be one of: 'position', 'binary', 'time', or 'text'.\n"
            "2. If category is 'position', provide [x, y, z] and orientation (theta).\n"
            "3. If category is 'time', provide minutes ago as a float.\n"
            "4. If category is 'duration', provide duration in minutes as a float.\n"
            "5. Always provide a text answer."
        )
        
        history.append(SystemMessage(content=formatting_instruction))
        
        try:
            structured_res = formatter.invoke(history)
            res_dict = structured_res.dict()
            
            # Ensure all keys required by AgentOutput are present
            required_keys = ['type', 'text', 'binary', 'position', 'orientation', 'duration', 'time']
            # Map 'type' if Pydantic model uses a different name, but here they match 'type'
            
            # Type normalization for AgentOutput dataclass
            output_dict = {
                'type': res_dict.get('type', 'text'),
                'text': res_dict.get('text', ''),
                'binary': res_dict.get('binary') or '',
                'position': res_dict.get('position') or [0.0, 0.0, 0.0],
                'orientation': res_dict.get('orientation') or 0.0,
                'duration': res_dict.get('duration') or 0.0,
                'time': res_dict.get('time') or 0.0
            }
            
            return AgentOutput.from_dict(output_dict)
        except Exception as e:
            print(f"Error during formatting: {e}")
            traceback.print_exc()
            raise e

if __name__ == "__main__":
    from remembr.memory.milvus_memory import MilvusMemory
    # Mock or local memory setup for testing
    try:
        memory = MilvusMemory("test", db_ip='127.0.0.1')
        agent = ReMEmbRAgent(llm_type='gpt-4o')
        agent.set_memory(memory)
        response = agent.query("Where can I sit?")
        print(response)
    except Exception as e:
        print(f"Could not run test main: {e}")
