from transformers import pipeline
import nltk
from nltk.chat.util import Chat, reflections

# Pre-trained conversational model from Hugging Face
chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")

def get_response(user_input):
    # Generate response from the model
    return chatbot(user_input)[0]['generated_text']

# Chatbot using nltk basic conversation (rule-based)
def nltk_chat():
    pairs = [
        [
            r"(.*) your name ?",
            ["My name is Chatbot, and I'm here to chat with you!"]
        ],
        [
            r"how are you ?",
            ["I'm doing great, how about you?"]
        ],
        [
            r"sorry (.*)",
            ["It's okay, no need to apologize."]
        ],
        [
            r"(hi|hello|hey)",
            ["Hello!", "Hi there!"]
        ],
        [
            r"quit",
            ["Bye, take care!"]
        ]
    ]
    
    nltk_bot = Chat(pairs, reflections)
    nltk_bot.converse()

def start_chat():
    print("Type 'nltk' for basic conversation or anything else for AI model conversation.")
    choice = input("Your choice: ").lower()

    if choice == "nltk":
        print("Starting basic conversation with NLTK chatbot...")
        nltk_chat()
    else:
        print("Starting conversation with AI model chatbot...")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
            response = get_response(user_input)
            print(f"AI: {response}")

if __name__ == "__main__":
    start_chat()
