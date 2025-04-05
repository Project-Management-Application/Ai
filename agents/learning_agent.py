# agents/learning_agent.py
from learning import UserLearningModule


class LearningAgent:
    def __init__(self):
        """
        Agent responsible for handling user corrections and learning
        """
        # Initialize the user learning module
        self.learning_module = UserLearningModule()

    def is_correction(self, query):
        """
        Check if the user is trying to correct a previous response

        Args:
            query (str): User's input

        Returns:
            bool: True if correction detected, False otherwise
        """
        return self.learning_module.is_correction(query)

    def handle_correction(self, query, last_query, last_response):
        """
        Process a user correction

        Args:
            query (str): User's correction message
            last_query (str): The previous user query
            last_response (str): The previous bot response

        Returns:
            str: Confirmation message
        """
        # Extract the corrected response from user input
        corrected_response = self.learning_module.extract_correction(query)

        # Add the correction to the learning module
        self.learning_module.add_user_correction(
            last_query,
            last_response,
            corrected_response
        )

        return "Thank you for the correction! I've stored this information for future improvement."

    def get_user_correction(self, query):
        """
        Check if there's a user correction available for this query

        Args:
            query (str): The user's query

        Returns:
            tuple: (response, is_correction) where is_correction indicates if the response
                  is from user corrections
        """
        correction, found = self.learning_module.find_matching_correction(query)

        if found:
            # Add attribution that this is based on user corrections
            return correction + " (based on users)", True

        return "", False