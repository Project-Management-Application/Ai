import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


class AdvancedQuestionGenerator:
    def __init__(self, model_name='google/flan-t5-base'):
        """
        Initialize advanced question generation model

        Args:
            model_name (str): Hugging Face model for question generation
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def generate_targeted_questions(self, topic, context, num_questions=3, max_length=128):
        """
        Generate targeted questions based on topic and context

        Args:
            topic (str): Topic name
            context (str): Detailed context of the topic
            num_questions (int): Number of questions to generate
            max_length (int): Maximum length of generated questions

        Returns:
            list: Generated questions with their specific context
        """
        # Create a more specific prompt that guides the model
        input_text = (
            f"You are an expert in {topic}. "
            f"Based on the following detailed context, "
            f"generate {num_questions} distinct and meaningful questions "
            f"that probe different aspects of this topic: {context}"
        )

        # Tokenize input
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        # Generate questions
        questions = []
        unique_questions = set()

        while len(questions) < num_questions:
            # Generate a single question
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7  # Add some creativity
            )

            # Decode question
            question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove duplicates and ensure question is not empty
            if question and question not in unique_questions:
                unique_questions.add(question)
                questions.append(question)

        # Generate specific answers for each question
        detailed_qa_pairs = []
        for question in questions:
            # Generate a targeted answer
            answer_prompt = (
                f"Context: {context}\n"
                f"Question: {question}\n"
                "Provide a concise and informative answer to this question."
            )

            # Tokenize answer prompt
            answer_inputs = self.tokenizer.encode(
                answer_prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)

            # Generate answer
            answer_outputs = self.model.generate(
                answer_inputs,
                max_length=200,
                num_return_sequences=1,
                do_sample=True
            )

            # Decode answer
            answer = self.tokenizer.decode(answer_outputs[0], skip_special_tokens=True)

            detailed_qa_pairs.append({
                "question": question,
                "answer": answer
            })

        return detailed_qa_pairs


def generate_and_append_topic_qa(knowledge_base_path, output_path, questions_per_topic=3):
    """
    Generate Q&A dataset and append to existing file

    Args:
        knowledge_base_path (str): Path to existing JSON knowledge base
        output_path (str): Path to save/append generated dataset
        questions_per_topic (int): Number of questions to generate per topic
    """
    # Try to load existing dataset, create empty dict if file doesn't exist
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_dataset = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_dataset = {}

    # Load knowledge base
    with open(knowledge_base_path, 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)

    # Initialize question generator
    question_generator = AdvancedQuestionGenerator()

    # Generate Q&A pairs for each topic
    for topic, contents in knowledge_base.items():
        # Combine contents if multiple exist
        context = " ".join(contents)

        try:
            # Generate targeted Q&A pairs
            qa_pairs = question_generator.generate_targeted_questions(
                topic,
                context,
                num_questions=questions_per_topic
            )

            # If topic doesn't exist in existing dataset, create it
            if topic not in existing_dataset:
                existing_dataset[topic] = {
                    "context": context,
                    "qa_pairs": []
                }

            # Append new Q&A pairs
            existing_dataset[topic]["qa_pairs"].extend(qa_pairs)

        except Exception as e:
            print(f"Error generating questions for {topic}: {e}")

    # Save updated dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(existing_dataset, f, ensure_ascii=False, indent=2)

    print(f"Appended Q&A pairs for {len(existing_dataset)} topics")

    # Print a few examples of newly added pairs
    print("\nSample Newly Added Q&A Pairs:")
    for topic, data in list(existing_dataset.items())[:2]:
        print(f"\nTopic: {topic}")
        for qa_pair in data['qa_pairs'][-questions_per_topic:]:
            print("Question:", qa_pair['question'])
            print("Answer:", qa_pair['answer'][:200] + "...")
            print("---")

    return existing_dataset


def main():
    # Process Scrum data
    generate_and_append_topic_qa(
        'Data/scrum_spacy_segmented.json',
        'Data/scrum_topic_qa_dataset.json'
    )

    # Process Kanban data
    generate_and_append_topic_qa(
        'Data/kanban_spacy_segmented.json',
        'Data/kanban_topic_qa_dataset.json'
    )

    # Process Project Management data
    generate_and_append_topic_qa(
        'Data/project_management_spacy_segmented.json',
        'Data/project_management_topic_qa_dataset.json'
    )


if __name__ == "__main__":
    main()