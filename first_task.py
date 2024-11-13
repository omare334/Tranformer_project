import torch
import wandb

wandb.init(project="simple_decoder", name='simple_decoder_overfit')

class MultiHeadAttentionLayer(torch.nn.Module):
    def __init__(self, emb_dim, num_heads, hidden_dim_ff):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads  # Dimension per head
        assert emb_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        
        self.linear_q = torch.nn.Linear(emb_dim, emb_dim)
        self.linear_k = torch.nn.Linear(emb_dim, emb_dim)
        self.linear_v = torch.nn.Linear(emb_dim, emb_dim)
        
        # Learnable bias for attention
        self.attn_embedding_bias = torch.nn.Parameter(torch.zeros(emb_dim))
        
        # Feedforward layer (two linear layers with ReLU in between)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, hidden_dim_ff),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim_ff, emb_dim)
        )

    def forward(self, emb):
        # Fix: Get dimensions correctly using size()
        seq_len = emb.size(0)
        # sinusodial positional encoding


        # Transform embeddings for query, key, and value, then reshape for multi-head attention
        query = self.linear_q(emb).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        key = self.linear_k(emb).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        value = self.linear_v(emb).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)

        # Calculate attention scores and apply softmax
        scaling_factor = self.head_dim ** 0.5
        similarity_matrix = torch.matmul(query, key.transpose(-2, -1)) / scaling_factor

        # Apply upper triangular mask (if required for causality)
        mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1) * -1e9
        similarity_matrix = similarity_matrix + mask

        # Apply softmax to get attention weights
        soft_matrix = torch.softmax(similarity_matrix, dim=-1)
    
        # Apply attention weights to values and reshape back
        attention = torch.matmul(soft_matrix, value)
        attention = attention.transpose(0, 1).contiguous()
        attention = attention.view(seq_len, -1)  # Reshape

        # Apply feedforward layer
        output = self.feedforward(attention)
        
        return output


class StackedAttentionModel(torch.nn.Module):
    def __init__(self, voc_size, emb_dim, num_heads, num_layers, hidden_dim_ff):
        super().__init__()
        
        self.num_layers = num_layers
        self.emb = torch.nn.Embedding(num_embeddings=voc_size, embedding_dim=emb_dim)
        self.ffw = torch.nn.Linear(emb_dim, voc_size, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)
        
        # Create a list of attention layers
        self.attn_layers = torch.nn.ModuleList([
            MultiHeadAttentionLayer(emb_dim, num_heads, hidden_dim_ff) 
            for _ in range(num_layers)
        ])

    def forward(self, inpt):
        emb = self.emb(inpt)  # Shape: [batch_size, seq_len, emb_dim]

        # Pass through the stacked attention layers
        for attn_layer in self.attn_layers:
            emb = attn_layer(emb)  # Update embeddings with the output of the attention layer
        
        # After passing through all attention layers, apply feedforward layer
        out = self.ffw(emb)  # Shape: [batch_size, seq_len, voc_size]
        # out = self.softmax(out)
        
        return out


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the sequence and target
sequence = ["<s>", "A", "A", "B", "B", "C", "C", "<e>"] *5  # The repeating pattern
vocab = {"<s>": 0, "A": 1, "B": 2, "C": 3, "<e>": 4}  # Define the vocabulary
voc_size = len(vocab)

tokens = [vocab[token] for token in sequence]  # Convert sequence to indices
targets = tokens[1:] + [tokens[0]]  # Shift targets to match the next-token prediction

# Convert sequence and targets to tensors
inpt = torch.LongTensor(tokens).to(device)
true_index = torch.LongTensor(targets).to(device)

# Model parameters
args = (voc_size, 64, 4, 1, 4)  # Example arguments (vocab size, embedding size, num heads)
model = StackedAttentionModel(*args)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

learning_rate = 0.0001

for epoch in range(10000):  # Train for multiple epochs to overfit
    model.train()  # Set model to training mode

    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    out = model(inpt)  # Model output, expected shape [seq_len, vocab_size]
    
    # Calculate the loss
    loss = criterion(out, true_index)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    # Log the loss
    wandb.log({'loss': loss.item(), 'learning_rate': learning_rate})

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# After training, generate a sequence
model.eval()
with torch.no_grad():
    # Start with first token
    test_input = torch.LongTensor([vocab["<s>"]]).to(device)
    generated = []
    
    # Generate one token at a time
    for _ in range(20):  # Generate 20 tokens
        output = model(test_input)
        next_token_id = torch.argmax(output[-1]).item()
        generated.append(next_token_id)
        test_input = torch.cat([test_input, torch.LongTensor([next_token_id]).to(device)])

# Convert ids back to tokens and print
generated_sequence = [k for id in generated for k, v in vocab.items() if v == id]
print("\nOriginal sequence:", " ".join(sequence))
print("Generated sequence:", " ".join(generated_sequence))

    
# Save the model's state dict
torch.save(model.state_dict(), 'model_overfit.pth')
wandb.finish()

