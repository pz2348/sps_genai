
import torch
import torch.nn as nn


# ==========================================
# LSTM Neural Network
# ==========================================

class LSTMNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])


# ==========================================
# LSTM Text Generation Model
# ==========================================

class LSTMTextGenerator:
    def __init__(self, corpus):
        self.words = " ".join(corpus).split()

        self.vocab = sorted(set(self.words))
        self.word_to_idx = {
            word: i for i, word in enumerate(self.vocab)
        }
        self.idx_to_word = {
            i: word for word, i in self.word_to_idx.items()
        }

        self.model = LSTMNetwork(len(self.vocab))

    def train(self, epochs=100):
        if len(self.words) < 2:
            raise ValueError("Corpus needs at least two words.")

        inputs = torch.tensor([
            self.word_to_idx[word]
            for word in self.words[:-1]
        ], dtype=torch.long).unsqueeze(1)

        targets = torch.tensor([
            self.word_to_idx[word]
            for word in self.words[1:]
        ], dtype=torch.long)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.01
        )
        criterion = nn.CrossEntropyLoss()

        self.model.train()

        for epoch in range(epochs):
            optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

        self.model.eval()
        print(f"LSTM training finished. Loss: {loss.item():.4f}")

    def generate_text(self, start_word, length=10):
        if length < 1:
            raise ValueError("Length must be at least 1.")

        if start_word not in self.word_to_idx:
            raise ValueError(
                f"Start word '{start_word}' is not in vocabulary."
            )

        generated = [start_word]
        current_word = start_word

        self.model.eval()

        with torch.no_grad():
            for _ in range(length):
                input_tensor = torch.tensor(
                    [[self.word_to_idx[current_word]]],
                    dtype=torch.long
                )

                logits = self.model(input_tensor)
                next_idx = torch.argmax(logits, dim=1).item()

                current_word = self.idx_to_word[next_idx]
                generated.append(current_word)

        return " ".join(generated)
