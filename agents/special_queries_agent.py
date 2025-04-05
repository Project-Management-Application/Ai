# agents/special_queries_agent.py
import numpy as np


class SpecialQueriesAgent:
    def __init__(self):
        """
        Agent responsible for handling special queries like greetings
        """
        # Predefined responses for special queries
        self.special_responses = {
            'greetings': [
                "Hello! I'm your Scrum and Kanban Assistant. How can I help you today?",
                "Hi there! Ready to discuss Agile methodologies?",
                "Welcome! I'm here to answer your questions about Scrum and Kanban.",
                "Greetings! What would you like to know about Agile practices?"
            ],
            'name': "I'm es2elni chat model, a Scrum and Kanban knowledge assistant.",
            'purpose': "I'm designed to help you understand and learn about Scrum and Kanban methodologies.",
            'creator': "I was created by Mimouni Med Aziz.",
            'daddy': 'Mimouni Med Aziz'
        }

    def process(self, query):
        """
        Handle special queries like greetings, name, purpose, creator

        Args:
            query (str): User's input query

        Returns:
            str: Special response or None
        """
        # Convert query to lowercase for easier matching
        query_lower = query.lower().strip()

        # Greeting patterns
        greeting_patterns = ['hi', 'hello', 'hey', 'greetings', 'hola', 'bonjour']

        # Check for greetings
        if any(pattern in query_lower for pattern in greeting_patterns):
            return np.random.choice(self.special_responses['greetings'])

        # Check for specific questions
        if 'your name' in query_lower or 'who are you' in query_lower:
            return self.special_responses['name']

        if 'your purpose' in query_lower or 'what do you do' in query_lower:
            return self.special_responses['purpose']

        if 'who made you' in query_lower or 'your creator' in query_lower:
            return self.special_responses['creator']

        if 'who is your daddy' in query_lower or 'daddy' in query_lower:
            return self.special_responses['daddy']

        return None
