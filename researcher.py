from dotenv import load_dotenv
from typing import  List,Annotated
from langgraph.graph import StateGraph,add_messages,END
from langchain_core.messages import  SystemMessage
from pydantic_settings import BaseModel
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch,TavilyExtract
from langchain.tools import tool
load_dotenv()
SystemPrompt="You are the helpful research assitant use the leverage tools to research and get the details and present them in a strcutre and markdown manner dont answer yourself use the tools to answer the question"

class ResearcherReport(BaseModel):
    messages : str
    report : str
class ResearcherState(BaseModel):
    messages:Annotated[list,add_messages] = []
# tavily_search_tool =  TavilySearch(
#     max_result  = 5
#     topics = "general"
# )
@tool
def search_web(query:str,no_result : int = 3):
    """search the web and return the query"""
    tavily_search_tool = TavilySearch(max_results=min(no_result,3),topics="general")
    search_result = tavily_search_tool.invoke(input = {"query":query})
    processed_results = {
        "query": query,
        "results" : []
    }
    for search in search_result["results"]:
        processed_results["results"].append({
            "title":search["title"],
            "url": search["url"],
            "content_preview": search["content"]
        })
    return processed_results

@tool
def extract_web_page(urls:List[str]):
    web_extract = TavilyExtract()
    results = web_extract.invoke(input ={"urls":urls})["results"]
    return results
llm = ChatOpenAI(model="gpt-4o-mini")


tools = [
    search_web,
    extract_web_page
]
llm_with_tools = llm.bind_tools(tools)
    
def researcher(state:Researcher):
    response llm_with_tools.invoke([
        SystemMessage(content =SystemPrompt)
    ] + state.messages)
    return {"messages": [response]}
def research_router(state:ResearcherState)-> str:
    if state.messages[-1].tool_calls:
        return "tools"
    return END