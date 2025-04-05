# agents/generation_agent.py
import requests
import json


class GenerationAgent:
    def __init__(self, api_key_path="api_key.txt", model_id="mistralai/Mistral-7B-Instruct-v0.1"):
        """
        Agent responsible for generating responses using Hugging Face API for Mistral

        Args:
            api_key_path (str): Path to file containing the Hugging Face API key
            model_id (str): Hugging Face model ID
        """
        # Load API key
        try:
            with open(api_key_path, 'r') as f:
                self.api_key = f.read().strip()
                print("Successfully loaded Hugging Face API key")
        except FileNotFoundError:
            raise FileNotFoundError(f"API key file not found at {api_key_path}")

        # Hugging Face API endpoint
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def _call_huggingface_api(self, payload):
        """
        Call the Hugging Face API

        Args:
            payload (dict): API request payload

        Returns:
            str: Generated text
        """
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors

            result = response.json()

            # Extract the generated text
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            elif "generated_text" in result:
                return result["generated_text"]
            else:
                print(f"Unexpected API response format: {result}")
                return "Error: Unexpected API response format"

        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return f"Error calling Hugging Face API: {str(e)}"

    def generate(self, query, similar_results, temperature=0.7):
        """
        Generate a response using the Hugging Face API for Mistral

        Args:
            query (str): User's query
            similar_results (list): List of similar results from retrieval
            temperature (float): Temperature for generation

        Returns:
            str: Generated response
        """
        # Construct context from similar results
        context_parts = []
        for result in similar_results:
            context_parts.append(f"Related Question: {', '.join(result['questions'])}")
            context_parts.append(f"Context information: {result['context']}")
            context_parts.append(f"Related methodologies: {', '.join(result['methodologies'])}")
            context_parts.append(f"Retrieved answer: {result['answer']}")

        context = "\n\n".join(context_parts)

        # Prepare prompt for generation
        prompt = f"""<s>[INST] You are an expert in project management, particularly in Scrum, Kanban, and other agile methodologies.
I will provide you with a user question and relevant information from my knowledge base.
Please provide a comprehensive, accurate answer by enhancing the retrieved information with your own knowledge.
Make sure your answer is precise, well-structured, and directly addresses the question.

USER QUESTION: {query}

RETRIEVED INFORMATION:
{context}

Please improve upon the retrieved answers by adding additional context, explanations, or examples as needed. Consolidate overlapping information and ensure the response is coherent and helpful. [/INST]"""

        # Call API
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1024,
                "temperature": temperature,
                "top_p": 0.95,
                "do_sample": True
            }
        }

        response = self._call_huggingface_api(payload)

        # Extract only the model's response (remove the prompt)
        if prompt in response:
            response = response.replace(prompt, "").strip()

        return response

    def generate_fallback(self, query, temperature=0.7):
        """
        Generate a fallback response when no good retrieval results exist

        Args:
            query (str): User's query
            temperature (float): Temperature for generation

        Returns:
            str: Generated response
        """
        # Construct prompt for Mistral
        prompt = f"""<s>[INST] You are an expert in project management, particularly in Scrum, Kanban, and other agile methodologies.
Please provide a comprehensive, accurate answer to the following question:

{query}

Provide a clear, well-structured response with examples if appropriate. [/INST]"""

        # Call API
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1024,
                "temperature": temperature,
                "top_p": 0.95,
                "do_sample": True
            }
        }

        response = self._call_huggingface_api(payload)

        # Extract only the model's response (remove the prompt)
        if prompt in response:
            response = response.replace(prompt, "").strip()

        return response