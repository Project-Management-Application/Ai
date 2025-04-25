import requests
import json


class GenerationAgent:
    def __init__(self, model_id="mistral"):
        """
        Agent responsible for generating responses using Ollama's local Mistral model

        Args:
            model_id (str): Ollama model ID (default: mistral, referring to mistral:7b-instruct)
        """
        # Ollama API endpoint (localhost)
        self.api_url = "http://localhost:11434/api/generate"
        self.model_id = model_id

    def _call_ollama_api(self, payload):
        """
        Call the Ollama API with streaming support

        Args:
            payload (dict): API request payload

        Returns:
            str: Generated text
        """
        try:
            # Enable streaming by setting stream=True
            response = requests.post(self.api_url, json=payload, stream=True)
            response.raise_for_status()  # Raise exception for HTTP errors

            full_response = ""
            for line in response.iter_lines():
                if line:
                    result = json.loads(line.decode('utf-8'))
                    if "response" in result:
                        full_response += result["response"]
                    if result.get("done", False):
                        break
            return full_response.strip()

        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return f"Error calling Ollama API: {str(e)}"

    def generate(self, query, similar_results, temperature=0.7):
        """
        Generate a response using the Ollama Mistral model

        Args:
            query (str): User's query
            similar_results (list): List of similar results from retrieval
            temperature (float): Temperature for generation

        Returns:
            str: Generated response
        """
        # Construct context from similar_results
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
            "model": self.model_id,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": 1024,
            "top_p": 0.95
            # Streaming enabled by default (no "stream": False)
        }

        response = self._call_ollama_api(payload)

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
            "model": self.model_id,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": 1024,
            "top_p": 0.95
            # Streaming enabled by default
        }

        response = self._call_ollama_api(payload)

        return response