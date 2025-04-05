import re
import json


def extract_sections(text):
    """
    Precisely segment the Scrum document into distinct sections
    """
    # Cleaned and organized sections
    sections = {
        'overview': [],
        'principles': [],
        'roles': [],
        'artifacts': [],
        'events': [],
        'practices': [],
        'challenges': [],
        'advanced_topics': []
    }

    # Keywords for identifying sections
    section_keywords = {
        'overview': ['what is scrum', 'definition', 'introduction', 'overview', 'background'],
        'principles': ['principles', 'core values', 'fundamental', 'philosophy', 'values'],
        'roles': ['scrum team', 'product owner', 'scrum master', 'development team', 'team roles', 'responsibilities'],
        'artifacts': ['product backlog', 'sprint backlog', 'increment', 'artifact', 'backlog', 'burndown'],
        'events': ['sprint', 'sprint planning', 'daily scrum', 'sprint review', 'sprint retrospective', 'event',
                   'ceremony'],
        'practices': ['implementation', 'how to', 'best practice', 'technique', 'approach'],
        'challenges': ['common problem', 'challenge', 'difficulty', 'pitfall', 'limitation'],
        'advanced_topics': ['scaling', 'enterprise', 'large scale', 'advanced', 'complex']
    }

    # Split text into lines
    lines = text.split('\n')

    # Current section being processed
    current_section = None
    current_section_content = []

    def assign_to_section(content):
        """Helper function to assign content to appropriate section"""
        content = ' '.join(content).strip()
        if not content:
            return

        # Try to identify the most appropriate section
        best_section = 'overview'  # default
        max_keyword_matches = 0

        for section, keywords in section_keywords.items():
            keyword_matches = sum(1 for keyword in keywords if keyword.lower() in content.lower())
            if keyword_matches > max_keyword_matches:
                best_section = section
                max_keyword_matches = keyword_matches

        # Add to the appropriate section
        sections[best_section].append(content)

    # Process each line
    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Check if line indicates a new section
        is_section_header = any(
            re.search(rf'\b{re.escape(keyword)}\b', line, re.IGNORECASE)
            for keywords in section_keywords.values()
            for keyword in keywords
        )

        if is_section_header:
            # If we have collected content for a previous section, assign it
            if current_section_content:
                assign_to_section(current_section_content)
                current_section_content = []

        # Collect content
        current_section_content.append(line)

    # Process the last section
    if current_section_content:
        assign_to_section(current_section_content)

    # Remove empty sections and limit content length
    cleaned_sections = {}
    for section, content_list in sections.items():
        # Remove very short or duplicate entries
        unique_content = list(dict.fromkeys(
            [c for c in content_list if len(c) > 50]
        ))

        if unique_content:
            cleaned_sections[section] = unique_content[:10]  # Limit to 10 entries per section

    return cleaned_sections


def segment_scrum_document(input_file, output_file):
    """
    Main function to segment Scrum document
    """
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Segment the document
    segmented_content = extract_sections(content)

    # Write organized content to JSON
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(segmented_content, outfile, indent=2, ensure_ascii=False)

    return segmented_content


# Usage
input_file = 'Data/scrum_data.txt'
output_file = 'scrum_segmented_data.json'
result = segment_scrum_document(input_file, output_file)

# Print out the sections found
print("Scrum Document Segmentation Results:")
for section, content_list in result.items():
    print(f"- {section.upper()}: {len(content_list)} entries")
    # Print first snippet of each section
    if content_list:
        print(f"  First snippet: {content_list[0][:200]}...\n")