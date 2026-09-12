from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(
        default="What is the attention mechanism?",
        description="The search query to run against the document.",
    )


class Result(BaseModel):
    rank: int
    text: str
    label: str = Field(description="Score or metadata label specific to the retrieval method.")


class SearchResponse(BaseModel):
    method: str
    query: str
    results: list[Result]
    metadata: dict = Field(default_factory=dict, description="Extra info, e.g. rewritten query or hypothesis.")
