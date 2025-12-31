from typing import Annotated,List
from pydantic import BaseModel
from langgraph.graph import add_messages
import operator


class ResearcherReportmodel(BaseModel):
    topic: str
    report: str


class ResearcherState(BaseModel):
    messages:Annotated[list,add_messages] = []
    research_reports : Annotated[list,operator.add] = []
