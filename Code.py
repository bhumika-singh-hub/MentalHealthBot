import nltk
import random
import string
from nltk.chat.util import Chat, reflections
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

positive_responses = [
    "I'm glad you're feeling good. Keep it up!",
    "That's great to hear!",
    "Awesome! Remember to cherish the good moments.",
    "That is amazing!! I am happy for you."
]

neutral_responses = [
    "I'm here if you want to talk.",
    "Would you like to tell me more?",
    "Feel free to share anything that's on your mind."
    ]

negative_responses = [
    "I'm sorry you're feeling this way. You're not alone.",
    "It might help to talk about it. I'm here for you.",
    "That sounds tough. I'm listening.",
    "Take a deep breath — you're doing your best."
]

pairs = [
    ["hi|hello|hey", ["Hello, how are you feeling today?", "Hi there, what's on your mind?"]],
    ["i feel (.*)", ["Why do you feel %1?", "Do you often feel %1?", "What makes you feel %1?"]],
    ["i am feeling (.*)", ["I'm here for you. Why are you feeling %1?", "How long have you been feeling %1?"]],
    ["i'm (.*)", ["Why are you %1?", "Does being %1 affect your daily life?"]],
    ["because (.*)", ["That makes sense. Have you tried talking to someone about this?", "How does that make you feel?"]],
    ["(.*) sad (.*)", ["It's okay to feel sad. Want to talk more about it?", "Sadness is a part of life. Tell me what's going on."]],
    ["(.*) stressed (.*)", ["Stress can be tough. What’s causing it?", "Have you been able to take breaks or relax?"]],
    ["(.*) anxious (.*)", ["Anxiety is hard to deal with. Want to share what's worrying you?", "Talking can help. What's making you anxious?"]],
    ["quit", ["Take care. Remember, you're not alone.", "Goodbye. Be kind to yourself."]],
    ["(.*)", ["", ""]]  # fallback to sentiment
]

chat = Chat(pairs, reflections)

def get_sentiment_response(user_input):
    sentiment = sia.polarity_scores(user_input)
    compound = sentiment['compound']
    if compound >= 0.5:
        return random.choice(positive_responses)
    elif compound <= -0.3:
        return random.choice(negative_responses)
    else:
        return random.choice(neutral_responses)

def main():
    print("Mental Health Chatbot: Hi, I'm here to support you. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Mental Health Chatbot: Take care. Everything will be alright.")
            break
        response = chat.respond(user_input)
        if response is None or response.strip() == "":
            response = get_sentiment_response(user_input)
        print(f"Mental Health Chatbot: {response}")

main()
