from pydantic import BaseModel


class MyState(BaseModel):
    input: str
    chat_history: list = []
    tool_messages: list = []
    intermediate_steps: list = []
    messages: list = []
    output: str = None
