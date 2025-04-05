# agents/base_agent.py
class BaseAgent:
    def __init__(self, name):
        self.name = name

    def process(self, input_data):
        """Process input and return output"""
        raise NotImplementedError