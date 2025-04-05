import re
import json
import os


def parse_document(input_file):
    """
    Parse a document with topics marked by ** at the beginning and end.
    Allows multiple entries for the same topic.

    Args:
        input_file (str): Path to the input text file

    Returns:
        dict: A dictionary with topics as keys and lists of content
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Regular expression to find topics and their content
    # This will match ** Topic ** followed by content until the next topic or end of file
    pattern = r'\*{2}(.*?)\*{2}\s*(.*?)(?=\*{2}|\Z)'

    # Find all matches
    matches = re.findall(pattern, content, re.DOTALL)

    # Create a dictionary to store results
    parsed_document = {}

    for match in matches:
        topic = match[0].strip()
        content = match[1].strip()

        # Only add non-empty topics and content
        if topic and content:
            # If topic doesn't exist, create a new list
            if topic not in parsed_document:
                parsed_document[topic] = []

            # Append content to the list of contents for this topic
            parsed_document[topic].append(content)

    return parsed_document


def save_to_json(parsed_data, output_file):
    """
    Save parsed document to a JSON file.

    Args:
        parsed_data (dict): Parsed document dictionary
        output_file (str): Path to the output JSON file
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(parsed_data, file, indent=2, ensure_ascii=False)


def main():
    """
    Main function to process multiple documents and save to JSON.
    """
    # List of input and output file pairs
    files = [
        ('Data/scrum_data.txt', 'Data/scrum_spacy_segmented.json'),
        ('Data/kanban_data.txt', 'Data/kanban_spacy_segmented.json')
    ]

    # Process each file
    for input_file, output_file in files:
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Warning: Input file {input_file} not found. Skipping.")
            continue

        # Parse the document
        parsed_document = parse_document(input_file)

        # Save to JSON
        save_to_json(parsed_document, output_file)

        # Print summary
        print(f"\nProcessed {input_file}:")
        print(f"Unique topics: {len(parsed_document)}")
        for topic, contents in parsed_document.items():
            print(f"- {topic}: {len(contents)} entries")


# Run the main function
if __name__ == "__main__":
    main()