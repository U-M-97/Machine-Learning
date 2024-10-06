import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
# from small_dataset import small_dataset
from datasets import load_dataset
import pickle

ds = load_dataset("imdb", cache_dir="dataset_cache")

# Tokenization function
def tokenize(sentences):
    tokens = []
    for sentence, _ in sentences:
        # split by whitespace
        words = sentence.replace('.', '').split()
        tokens.append(words)
    return tokens

# tokenized_data = []
# for i in range(len(ds['train'])):
#     if i == 1000:
#         break
#     words = ds['train'][i]['text'].replace('.', '').split()
#     tokenized_data.append(words)

# with open("imdb.pkl", 'wb') as file:
#     pickle.dump(tokenized_data, file)

with open("imdb.pkl", 'rb') as file:
    tokenized_data = pickle.load(file)

def create_vocabulary(tokenized_data):
    vocabulary = set()
    for sentence in tokenized_data:
        vocabulary.update(sentence)
    return sorted(vocabulary)

vocabulary = create_vocabulary(tokenized_data)

def encode_sentences(tokenized_data, vocabulary):
    word_to_index = {word: index for index, word in enumerate(vocabulary)}
    encoded_data = []
    
    for sentence in tokenized_data:
        encoded_sentence = [word_to_index[word] for word in sentence]
        encoded_data.append(encoded_sentence)
    
    return encoded_data, word_to_index

encoded_data, word_to_index = encode_sentences(tokenized_data, vocabulary)

# Create input-output pairs for language modeling
def create_input_output_pairs(encoded_data):
    inputs, outputs = [], []
    for sentence in encoded_data:
        if(len(sentence) > 300):
            sentence = sentence[:300]

        for i in range(1, len(sentence)):
            inputs.append(sentence[:i])  # Input is the sentence up to the i-th token
            outputs.append(sentence[i])   # Output is the i-th token
    return inputs, outputs

inputs, outputs = create_input_output_pairs(encoded_data)

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MaskedMultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads  # Dimension per head
        
        # Linear layers for queries, keys, and values
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        
        # Linear layer to combine the heads' outputs
        self.fc_out = nn.Linear(embed_size, embed_size)
        
    def forward(self, x):
        batch_size, seq_len, embed_size = x.shape
        
        # Create queries, keys, and values
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Split into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for easier dot product calculation (batch_size, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_scores.masked_fill_(mask.unsqueeze(0).unsqueeze(1), float('-inf'))  # Apply mask
        attention_weights = F.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_weights, V)
        
        # Concatenate the attention output from all heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_size)
        
        # Pass through the final linear layer
        out = self.fc_out(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, embed_size, hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, hidden_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MaskedMultiHeadAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.ff = FeedForward(embed_size, hidden_dim)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Attention + Residual connection + Layer Norm
        attention_output = self.attention(x)
        x = self.norm1(x + attention_output)
        x = self.dropout(x)
        
        # Feedforward + Residual connection + Layer Norm
        feedforward_output = self.ff(x)
        x = self.norm2(x + feedforward_output)
        x = self.dropout(x)
        
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, seq_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pos_enc = torch.zeros(seq_len, embed_size)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * -(torch.log(torch.tensor(10000.0)) / embed_size))
        
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pos_enc', pos_enc.unsqueeze(0))  # Use register_buffer for non-learnable parameters
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pos_enc[:, :seq_len]
    
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, seq_len=100, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, seq_len)
        
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, hidden_dim, dropout) 
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return x
    
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, seq_len=100, dropout=0.1):
        super(LanguageModel, self).__init__()
        self.encoder = TransformerEncoder(vocab_size, embed_size, num_heads, hidden_dim, num_layers, seq_len, dropout)
        self.fc_out = nn.Linear(embed_size, vocab_size)
    
    def forward(self, x):
        encoder_output = self.encoder(x)
        logits = self.fc_out(encoder_output)
        return logits

# Hyperparameters
batch_size = 1
# seq_len = max(len(sentence) for sentence in tokenized_data)
seq_len = 300
embed_size = 512
num_heads = 8
vocab_size = len(vocabulary)  # Size of the vocabulary
print(vocab_size)
hidden_dim = 2048  # Dimensionality of the hidden layer in the feedforward network
num_layers = 6  # Number of transformer layers
dropout = 0.1  # Dropout rate
learning_rate = 0.001
num_epochs = 10

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_seq = self.inputs[idx]
        output_seq = self.outputs[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(output_seq, dtype=torch.long)

# Define a custom collate function for dynamic padding
def collate_fn(batch):
    inputs, outputs = zip(*batch)
    
    padded_inputs = []
    padded_outputs = []
    
    for i in range(len(batch)):
        input_list = inputs[i].tolist()
        while len(input_list) < seq_len:
            input_list.append(0)
        padded_inputs.append(torch.tensor(input_list))

        output_list = [outputs[i].tolist()]
        while len(output_list) < seq_len:
            output_list.append(-1)
        padded_outputs.append(torch.tensor(output_list))

    padded_inputs = torch.stack(padded_inputs)
    padded_outputs = torch.stack(padded_outputs)

    return padded_inputs, padded_outputs

# Create Dataset and DataLoader
dataset = CustomDataset(inputs, outputs)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
print(len(dataset))

# Create the language model
model = LanguageModel(vocab_size, embed_size, num_heads, hidden_dim, num_layers, seq_len=seq_len, dropout=dropout)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padded values in output
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
model.train()

def train():
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (batch_inputs, batch_outputs) in enumerate(dataloader):
            batch_inputs, batch_outputs = batch_inputs.to(device), batch_outputs.to(device)

            # Forward pass
            optimizer.zero_grad()
            
            # Get model predictions
            logits = model(batch_inputs)  # Shape: [batch_size, seq_len, vocab_size]
            
            # Reshape for the loss function
            logits = logits.view(-1, vocab_size)  # Flatten the predictions: [batch_size * seq_len, vocab_size]
            batch_outputs = batch_outputs.view(-1)  # Flatten the output: [batch_size * seq_len]
            
            # Compute the loss
            loss = loss_fn(logits, batch_outputs)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 1000 == 0:
                print(batch_idx)
                print("total loss = ", total_loss)
            
        # Print average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    torch.save(model.state_dict(), "p.pt")
    
model.load_state_dict(torch.load("p.pt"))

# Reverse the word-to-index mapping
index_to_word = {index: word for word, index in word_to_index.items()}

def generate_text(model, word_to_index, index_to_word, seed_text, max_len=50, temperature=1.0):
    model.eval()  # Set the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tokenize the seed text
    seed_tokens = [word_to_index[word] for word in seed_text.split() if word in word_to_index]
    
    # Initialize the input with the seed text
    input_seq = torch.tensor(seed_tokens, dtype=torch.long).unsqueeze(0).to(device)  # Shape: [1, len(seed_tokens)]
 
    generated_tokens = seed_tokens.copy()
    
    # Generate text by predicting the next word iteratively
    for _ in range(max_len):
        with torch.no_grad():
            # Pass the input through the model
            logits = model(input_seq)
            logits = logits[:, -1, :]  # Get the logits of the last token in the sequence
            
            # Apply temperature to logits
            logits = logits / temperature
            
            # Apply softmax to get probabilities and sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            # Append the next token to the generated sequence
            generated_tokens.append(next_token_id)
            
            # Update the input sequence with the new token
            input_seq = torch.cat([input_seq, torch.tensor([[next_token_id]], dtype=torch.long).to(device)], dim=1)
    
    # Convert token ids to words
    generated_text = ' '.join([index_to_word[token] for token in generated_tokens])
    
    return generated_text

# Example usage
seed_text = "The movie was"
generated_text = generate_text(model, word_to_index, index_to_word, seed_text, max_len=297, temperature=1.0)
print("Generated Text: ", generated_text)