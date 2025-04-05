import json
import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# MongoDB connection settings
uri = "mongodb+srv://medazizmimouni35:mzZMY4yDSj2XMbfu@chatbotfed.7xtvyf7.mongodb.net/?retryWrites=true&w=majority&appName=ChatbotFed"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Choose the database
db = client['chatbot_qa_database']

# Test connection
try:
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
except Exception as e:
    print(f"Connection error: {e}")
    exit(1)

# Define the data directory (one level up from the current script)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, 'data')

print(f"Looking for data files in: {data_dir}")

# Verify data directory exists
if not os.path.exists(data_dir):
    print(f"Error: Data directory not found at {data_dir}")
    exit(1)

# Get list of JSON files
json_files = [f for f in os.listdir(data_dir) if f.endswith('_dataset.json')]

if not json_files:
    print(f"No JSON files found in {data_dir} directory")
    exit(1)
else:
    print(f"Found files: {json_files}")

# Process each JSON file
for file_name in json_files:
    # Extract collection name from the file name (remove _dataset.json)
    collection_name = file_name.replace('_dataset.json', '')

    # Create or get collection
    collection = db[collection_name]

    # Full path to the file
    file_path = os.path.join(data_dir, file_name)

    try:
        # Read JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Insert data
        if isinstance(data, list):
            # If data is a list of documents
            if data:
                result = collection.insert_many(data)
                print(
                    f"Inserted {len(result.inserted_ids)} documents from {file_name} into {collection_name} collection")
            else:
                print(f"No data found in {file_name}")
        elif isinstance(data, dict):
            # If data is a single document
            result = collection.insert_one(data)
            print(f"Inserted document from {file_name} into {collection_name} collection with ID: {result.inserted_id}")
        else:
            print(f"Unsupported data format in {file_name}")

    except json.JSONDecodeError:
        print(f"Error: {file_name} contains invalid JSON")
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

print("Import completed!")
