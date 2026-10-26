
from app.lstm_model import LSTMTextGenerator


# ==========================================
# Sample Training Corpus
# ==========================================

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas",
    "The Count of Monte Cristo tells the story of Edmond Dantes",
    "The story is about revenge and justice",
    "The novel is a famous adventure story",
    "The Count is a character in the novel"
]


# ==========================================
# Create LSTM Model
# ==========================================

lstm_model = LSTMTextGenerator(corpus)


# ==========================================
# Train LSTM
# ==========================================

print("Starting LSTM training...")

lstm_model.train(epochs=100)


# ==========================================
# Generate Text
# ==========================================

generated_text = lstm_model.generate_text(
    start_word="The",
    length=10
)

print("Generated Text:")
print(generated_text)
