from agents.retrieval_agent import RetrievalAgent
from agents.generation_agent import GenerationAgent
from agents.special_queries_agent import SpecialQueriesAgent
from agents.learning_agent import LearningAgent
from agents.conversation.conversational_agent import MistralConversationAgent


class OrchestratorAgent:
    def __init__(self, mistral_api_key=None):
        """
        Orchestrator agent that coordinates between other agents

        Args:
            mistral_api_key (str, optional): API key for Hugging Face to use with MistralConversationAgent
        """
        self.retrieval_agent = RetrievalAgent()
        self.generation_agent = GenerationAgent()
        self.special_queries_agent = SpecialQueriesAgent()
        self.learning_agent = LearningAgent()

        # Initialize the conversation agent if API key is provided
        if mistral_api_key:
            self.conversation_agent = MistralConversationAgent(mistral_api_key)
        else:
            # Try to load API key from file
            try:
                from agents.conversation.conversational_agent import load_api_key
                api_key = load_api_key()
                self.conversation_agent = MistralConversationAgent(api_key)
            except Exception as e:
                print(f"Could not initialize conversation agent: {str(e)}")
                self.conversation_agent = None

        # Track conversation history
        self.last_query = None
        self.last_response = None

        # Conversation mode flag
        self.conversation_mode = False

    def is_conversation_request(self, query):
        """
        Determine if a query is requesting a conversation rather than information

        Args:
            query (str): The user's query

        Returns:
            bool: True if this appears to be a conversational query
        """
        # Simple heuristic - can be improved with more sophisticated NLP
        conversation_indicators = [
            "let's chat", "talk with me", "have a conversation",
            "let's talk", "just chat", "casual conversation",
            "talk about", "what do you think about"
        ]

        query_lower = query.lower()

        # Check if query contains conversation indicators
        for indicator in conversation_indicators:
            if indicator in query_lower:
                return True

        # Check if already in conversation mode and query is conversational
        if self.conversation_mode and len(query.split()) < 10 and "?" not in query:
            return True

        return False

    def is_out_of_domain(self, query):
        """
        Determine if a query is likely outside the domain of our QA dataset
        or just a casual/conversational input

        Args:
            query (str): The user's query

        Returns:
            bool: True if this appears to be out of domain
        """
        # Check for very short queries
        if len(query.split()) <= 3:
            return True

        # Check for common conversational phrases
        casual_phrases = [
            "hello", "hi", "hey", "yoo", "what's up", "how are you",
            "what?", "huh?", "ok", "nice", "thanks", "thank you"
        ]

        query_lower = query.lower()
        for phrase in casual_phrases:
            if query_lower.startswith(phrase) or query_lower == phrase:
                return True

        # You could also check if the query contains any domain-related keywords
        domain_keywords = ["scrum", "kanban", "agile", "project", "sprint",
                           "backlog", "methodology", "team", "product owner"]

        # If none of the domain keywords are present, likely out of domain
        if not any(keyword in query_lower for keyword in domain_keywords):
            return True

        return False


    def toggle_conversation_mode(self, value=None):
        """
        Toggle or set the conversation mode

        Args:
            value (bool, optional): If provided, set mode to this value
        """
        if value is not None:
            self.conversation_mode = value
        else:
            self.conversation_mode = not self.conversation_mode

        # Reset conversation history when exiting conversation mode
        if not self.conversation_mode and self.conversation_agent:
            self.conversation_agent.reset_conversation()

        return self.conversation_mode

    def process_query(self, query):
        """
        Process a user query by routing to appropriate agent

        Args:
            query (str): User's input

        Returns:
            str: Response to the user
        """
        # Check for explicit conversation mode commands
        if query.lower() in ["start conversation", "begin chat"]:
            self.toggle_conversation_mode(True)
            return "I'm now in conversation mode. Feel free to chat with me casually. Say 'end conversation' when you want to return to regular query mode."

        if query.lower() in ["end conversation", "stop chat"]:
            self.toggle_conversation_mode(False)
            return "Conversation mode ended. I'm back to regular query processing mode."

        # Handle conversation mode or conversation requests
        if (self.conversation_mode or self.is_conversation_request(query)) and self.conversation_agent:
            if not self.conversation_mode:
                self.toggle_conversation_mode(True)
                prefix = "I'll switch to conversation mode. "
            else:
                prefix = ""

            response = self.conversation_agent.chat(query)
            self.last_query = query
            self.last_response = response
            return prefix + response

        # Check if this is a correction to a previous response
        if self.learning_agent.is_correction(query):
            response = self.learning_agent.handle_correction(
                query, self.last_query, self.last_response
            )
            return response

        # Check if we have a user correction for this query
        user_correction, is_user_correction = self.learning_agent.get_user_correction(query)

        if is_user_correction:
            # We have a user-corrected answer for this query, use it
            self.last_query = query
            self.last_response = user_correction
            return user_correction

        # After checking for special queries
        special_response = self.special_queries_agent.process(query)
        if special_response:
            # Store the interaction for potential future correction
            self.last_query = query
            self.last_response = special_response
            return special_response

        # Check if query is out of domain or just conversational
        if self.is_out_of_domain(query):
            # Skip retrieval and go straight to Mistral
            response = self.generation_agent.generate_fallback(query)
            self.last_query = query
            self.last_response = response
            return response

        # Find similar questions and contexts (only if we think it's an in-domain query)
        similar_results = self.retrieval_agent.retrieve(query)
        # UPDATED LOGIC: Always use generation agent to enhance answers
        if similar_results:
            # If we have good retrieval results, use them with the generation agent
            response = self.generation_agent.generate(query, similar_results)

            # Add methodology context if we have a particularly good match
            if similar_results[0]['distance'] < self.retrieval_agent.SIMILARITY_THRESHOLD:
                best_match = similar_results[0]
                methodologies = best_match['methodologies']
                if methodologies:
                    methodologies_str = " and ".join(methodologies)
                    response += f"\n\n(Information based on {methodologies_str} methodologies)"
        else:
            # No retrieval results, use fallback generation
            response = self.generation_agent.generate_fallback(query)

        # Store the interaction for potential future correction
        self.last_query = query
        self.last_response = response

        return response