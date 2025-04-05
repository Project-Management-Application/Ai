import json
import os
import numpy as np


class UserLearningModule:
    def __init__(self, learning_dataset_path='Data/chatbot_learning.json'):
        """
        Initialize User Learning Module

        Args:
            learning_dataset_path (str): Path to the user-generated learning dataset
        """
        self.learning_dataset_path = learning_dataset_path
        self.learning_dataset = self.load_or_create_learning_dataset()

    def load_or_create_learning_dataset(self):
        """
        Load existing learning dataset or create a new one if it doesn't exist

        Returns:
            dict: Learning dataset
        """
        try:
            with open(self.learning_dataset_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # Create a new learning dataset structure
            return {
                'corrections': []
            }

    def save_learning_dataset(self):
        """
        Save the current learning dataset to file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.learning_dataset_path), exist_ok=True)

        with open(self.learning_dataset_path, 'w', encoding='utf-8') as f:
            json.dump(self.learning_dataset, f, ensure_ascii=False, indent=4)

    def add_user_correction(self, original_query, original_response, corrected_response):
        """
        Add a user correction to the learning dataset

        Args:
            original_query (str): The original user query
            original_response (str): The chatbot's original response
            corrected_response (str): The user's suggested correct response
        """
        correction_entry = {
            'original_query': original_query,
            'original_response': original_response,
            'corrected_response': corrected_response,
            'timestamp': np.datetime64('now').astype(str)
        }

        # Add to corrections list
        self.learning_dataset['corrections'].append(correction_entry)

        # Save the updated learning dataset
        self.save_learning_dataset()

    def is_correction(self, user_input):
        """
        Determine if the user input is a correction

        Args:
            user_input (str): User's input text

        Returns:
            bool: True if input suggests a correction, False otherwise
        """
        correction_indicators = [
            'no,', 'no ', 'incorrect', 'that\'s wrong',
            'wrong', 'incorrect', 'not right', 'fix this'
        ]

        return any(indicator in user_input.lower() for indicator in correction_indicators)

    def extract_correction(self, user_input):
        """
        Extract the corrected response from user input

        Args:
            user_input (str): User's correction input

        Returns:
            str: Extracted corrected response
        """
        # Remove correction indicators
        for indicator in ['no,', 'no ', 'incorrect', 'that\'s wrong', 'wrong', 'not right', 'fix this']:
            if user_input.lower().startswith(indicator):
                return user_input[len(indicator):].strip()

        return user_input.strip()

    def find_matching_correction(self, query):
        """
        Find a matching correction for the given query

        Args:
            query (str): The user's current query

        Returns:
            tuple: (corrected_response, found) where found is True if a match was found
        """
        # Simple exact matching for now
        for correction in self.learning_dataset['corrections']:
            if correction['original_query'].lower() == query.lower():
                return correction['corrected_response'], True

        # Could implement more sophisticated matching (fuzzy matching, embeddings, etc.)

        return "", False
