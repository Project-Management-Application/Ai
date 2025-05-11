ScrumKanbanChatbot

Overview

This project, ScrumKanbanChatbot, is a Retrieval-Augmented Generation (RAG) AI system designed to assist with Scrum and Kanban project management. It processes text queries and UML use case diagrams to provide relevant responses and generate sprint backlogs. The system uses YOLOv5 and PaddleOCR for UML diagram processing, a retrieval agent for fetching relevant information, and a generation agent powered by Mistral for response generation. Metrics are computed to evaluate retrieval accuracy (Precision@5, MRR) and generation quality (BLEU, ROUGE).

Ignored Files and Folders

For security, privacy, and repository size considerations, the following files and folders are excluded from this GitHub repository (as specified in .gitignore):





data/: Contains datasets (scrum_topic_qa_dataset.json, kanban_topic_qa_dataset.json, project_management_topic_qa_dataset.json) used for retrieval and training. These are excluded due to their large size and potential sensitivity.



api_key.txt and **/api_key.txt: API keys for external services (e.g., Mistral API) are stored in these files and ignored to prevent accidental exposure.



database: Any local database files (e.g., for storing embeddings or logs) are excluded to avoid sharing sensitive or environment-specific data.



yolov5s.pt: Pre-trained YOLOv5 model weights for UML diagram detection. Excluded due to its large size.



yolov5/: The YOLOv5 repository or related files, ignored to keep the repository lightweight.



uml_dataset: Dataset of UML images used for training/testing, excluded due to size and privacy concerns.

These exclusions ensure the repository remains secure and manageable while allowing the core code to be shared. To run the project, you’ll need to provide these files separately as per the setup instructions (not included here).
