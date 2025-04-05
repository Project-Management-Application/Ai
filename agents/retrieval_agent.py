import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json


class RetrievalAgent:
    def __init__(self,
                 scrum_dataset_path='Data/scrum_topic_qa_dataset.json',
                 kanban_dataset_path='Data/kanban_topic_qa_dataset.json',
                 project_management_dataset_path='Data/project_management_topic_qa_dataset.json'):
        """
        Agent responsible for retrieving relevant information

        Args:
            scrum_dataset_path (str): Path to the Scrum QA dataset
            kanban_dataset_path (str): Path to the Kanban QA dataset
            project_management_dataset_path (str): Path to the Project Management QA dataset
        """
        # Load datasets
        self.dataset = {}
        for dataset_path, methodology in [
            (scrum_dataset_path, 'Scrum'),
            (kanban_dataset_path, 'Kanban'),
            (project_management_dataset_path, 'Project Management')
        ]:
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    self.dataset[methodology] = json.load(f)
            except FileNotFoundError:
                print(f"Warning: {methodology} dataset not found.")

        # Prepare embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Confidence threshold for matching
        self.SIMILARITY_THRESHOLD = 1.5  # Lower means stricter matching

        # Prepare embeddings and index
        self.prepare_embeddings()

    def prepare_embeddings(self):
        """
        Create embeddings for all questions and prepare Faiss index
        """
        self.questions = []
        self.answers = []
        self.contexts = []
        self.methodologies = []

        for methodology, methodology_data in self.dataset.items():
            for topic, topic_data in methodology_data.items():
                context = topic_data['context']
                for qa_pair in topic_data['qa_pairs']:
                    self.questions.append(qa_pair['question'])
                    self.answers.append(qa_pair['answer'])
                    self.contexts.append(context)
                    self.methodologies.append(methodology)

        # Create embeddings
        self.question_embeddings = self.embedding_model.encode(self.questions)

        # Create Faiss index
        dimension = self.question_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(self.question_embeddings).astype('float32'))

    def retrieve(self, query, top_k=5):
        """
        Find most similar questions to the user's query

        Args:
            query (str): User's input question
            top_k (int): Number of top similar questions to retrieve

        Returns:
            list: Top similar questions, answers, and contexts
        """
        # Embed query
        query_embedding = self.embedding_model.encode([query])

        # Search in Faiss index
        distances, indices = self.index.search(query_embedding, top_k)

        # Group duplicate answers
        results_dict = {}
        for idx in indices[0]:
            question = self.questions[idx]
            answer = self.answers[idx]
            context = self.contexts[idx]
            methodology = self.methodologies[idx]
            distance = distances[0][list(indices[0]).index(idx)]

            # Use answer as key to combine duplicate answers
            if answer not in results_dict:
                results_dict[answer] = {
                    'questions': [question],
                    'context': context,
                    'methodologies': [methodology],
                    'distances': [distance]
                }
            else:
                results_dict[answer]['questions'].append(question)
                results_dict[answer]['methodologies'].append(methodology)
                results_dict[answer]['distances'].append(distance)

        # Convert to list and sort
        results = []
        for answer, data in results_dict.items():
            results.append({
                'answer': answer,
                'questions': data['questions'],
                'context': data['context'],
                'methodologies': list(set(data['methodologies'])),
                'distance': min(data['distances'])
            })

        # Sort results by distance
        results.sort(key=lambda x: x['distance'])

        return results
