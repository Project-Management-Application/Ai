# agents/mistral_conversation_agent.py

import requests
import json
import os
from typing import List, Dict, Optional


class MistralConversationAgent:
    """
    A simple conversation agent using Mistral 7B Instruct v0.2 via Hugging Face API.
    """

    def __init__(self, api_key: str, model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initialize the conversation agent with your Hugging Face API key.

        Args:
            api_key: Your Hugging Face API token
            model_id: The model identifier (default: mistralai/Mistral-7B-Instruct-v0.2)
        """
        self.api_key = api_key
        self.model_id = model_id
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.conversation_history: List[Dict[str, str]] = []

    def format_prompt(self, user_input: str) -> str:
        """
        Format the conversation history and new user input into a prompt for Mistral.

        Args:
            user_input: The latest user message

        Returns:
            A formatted prompt string
        """
        # Build conversation context from history
        prompt = ""

        for message in self.conversation_history:
            if message["role"] == "user":
                prompt += f"<s>[INST] {message['content']} [/INST]\n"
            else:  # assistant
                prompt += f"{message['content']}</s>\n"

        # Add the current user message
        prompt += f"<s>[INST] {user_input} [/INST]\n"

        return prompt

    def query_model(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Send a request to the Hugging Face API with the given prompt.

        Args:
            prompt: The formatted conversation prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness (higher = more random)

        Returns:
            The model's response text
        """
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": True,
                "return_full_text": False
            }
        }

        response = requests.post(self.api_url, headers=self.headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "").strip()
            return "Error: Unexpected response format"
        else:
            return f"Error: API request failed with status code {response.status_code}"

    def chat(self, user_input: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Process a user message and get a response from the model.

        Args:
            user_input: The user's message
            max_tokens: Maximum number of tokens for the response
            temperature: Controls randomness of output

        Returns:
            The assistant's response
        """
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_input})

        # Format prompt with conversation history
        prompt = self.format_prompt(user_input)

        # Query the model
        response = self.query_model(prompt, max_tokens, temperature)

        # Add response to history
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def reset_conversation(self):
        """Clear the conversation history."""
        self.conversation_history = []


# Function to load API key from file
def load_api_key(file_path="api_key.txt"):
    """
    Load API key from a file.

    Args:
        file_path: Path to the file containing just the API key

    Returns:
        The API key as a string
    """
    try:
        with open(file_path, "r") as f:
            api_key = f.read().strip()
            return api_key
    except FileNotFoundError:
        print(f"API key file not found at {file_path}")
        api_key = input("Enter your Hugging Face API key: ")

        # Save the entered key for future use
        with open(file_path, "w") as f:
            f.write(api_key)

        print(f"API key saved to {file_path}")
        return api_key