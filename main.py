# main.py
from agents.orchestrator_agent import OrchestratorAgent


class ScrumKanbanChatbot:
    def __init__(self):
        """
        Initialize the agent-based Scrum and Kanban Chatbot
        """
        self.orchestrator = OrchestratorAgent()

    def generate_response(self, query):
        """
        Generate a response to the user query

        Args:
            query (str): User's input question

        Returns:
            str: Generated response
        """
        return self.orchestrator.process_query(query)

    def chat(self):
        """
        Interactive chat interface
        """
        print(
            "Scrum and Kanban Chatbot: Hello! I'm ready to answer your questions about Scrum and Kanban. Type 'exit' to end.")

        while True:
            query = input("You: ")

            if query.lower() == 'exit':
                print("Scrum and Kanban Chatbot: Goodbye!")
                break

            response = self.generate_response(query)
            print("Scrum and Kanban Chatbot:", response)
            print("\n")


def main():
    chatbot = ScrumKanbanChatbot()
    chatbot.chat()


if __name__ == "__main__":
    main()
