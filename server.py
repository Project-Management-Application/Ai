from flask import Flask, request, jsonify
from flask_cors import CORS
from main import ScrumKanbanChatbot  # Import chatbot class

app = Flask(__name__)
CORS(app)  # Enable CORS

# Initialize chatbot
chatbot = ScrumKanbanChatbot()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "Please enter a message."})

    # Get chatbot response
    bot_reply = chatbot.generate_response(user_message)

    return jsonify({"response": bot_reply})

if __name__ == '__main__':
    app.run(host='10.30.30.26', port=5000, debug=True)
