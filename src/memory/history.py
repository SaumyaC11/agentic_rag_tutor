from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage


class ChatHistoryBuffer:
    def __init__(self, max_turns=10):
        self.max_turns = max_turns
        self.buffer = []

    def append_turn(self, human_msg: str, ai_msg: str):
        self.buffer.append(HumanMessage(content=human_msg))
        self.buffer.append(AIMessage(content=ai_msg))
        if len(self.buffer) > self.max_turns * 2:
            self.buffer = self.buffer[-self.max_turns * 2:]

    def get(self):
        return self.buffer
