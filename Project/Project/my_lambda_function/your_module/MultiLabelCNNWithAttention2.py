import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelCNNWithAttention2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiLabelCNNWithAttention2, self).__init__()

        # Define layers
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.BatchNorm1d(32)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.BatchNorm1d(64)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.BatchNorm1d(128)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )

        self.attention = MultiHeadAttention(256)  # Attention layer to apply after conv layers
        
        self.dropout = nn.Dropout(p=0.4)

        # Calculate output size for the fully connected layer after conv layers
        fc_input_dim = 256 * (input_dim//8)  # Hardcode the correct dimension based on the output shape

        self.fc1 = nn.Linear(fc_input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for Conv1d input (batch, channels, length)
        
        # Apply conv layers with batch normalization and pooling
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.dropout(x)
        x = self.layer4(x)

        # Apply attention after the last convolution layer
        x = x.transpose(1, 2)  # (batch_size, seq_len, feature_dim)
        x = self.attention(x)
        x = x.transpose(1, 2)  # Transpose back to (batch_size, feature_dim, seq_len)

        # Flatten the tensor using reshape instead of view
        x = x.reshape(x.size(0), -1)

        # Dropout and fully connected layers
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return self.sigmoid(x)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.scale = (self.head_dim ** -0.5)
        
        self.qkv = nn.Linear(input_dim, input_dim * 3)
        self.fc_out = nn.Linear(input_dim, input_dim)
        
        self.norm = nn.LayerNorm(input_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.projection_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.norm(x)  # Apply layer normalization
        
        # Single projection for Q, K, V
        qkv = self.qkv(x)  # (batch_size, seq_len, 3 * input_dim)
        Q, K, V = qkv.chunk(3, dim=-1)  # Split into Q, K, V
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_scores = attention_scores - attention_scores.max(dim=-1, keepdim=True).values  # Stabilize
        attention_weights = self.attention_dropout(F.softmax(attention_scores, dim=-1))
        attended_values = torch.matmul(attention_weights, V)
        
        # Concatenate heads and pass through final linear layer
        attended_values = attended_values.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        out = self.projection_dropout(self.fc_out(attended_values))
        
        # Add residual connection
        return x + out
