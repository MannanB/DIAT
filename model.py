import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """x shape: [seq_len, batch_size, d_model]"""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# ---------------------
# Transformer Decoder
# ---------------------
class TransformerEncoder(nn.Module):
    """
    A small wrapper for a TransformerEncoder with an Embedding + PositionalEncoding.
    We allow passing a `src_key_padding_mask` for padded tokens.
    """
    def __init__(self, vocab_size, d_model, num_heads, ff_hidden_layer, dropout):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=ff_hidden_layer,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
    def forward(self, x, src_key_padding_mask=None):
        """
        x shape: [batch_size, seq_len]
        return shape: [batch_size, seq_len, d_model]
        """
        # Convert shape to [seq_len, batch_size]
        x = x.transpose(0, 1)  # -> [seq_len, batch_size]
        emb = self.embedding(x)  # -> [seq_len, batch_size, d_model]
        emb = self.pos_encoder(emb)  # -> [seq_len, batch_size, d_model]
        # Pass to transformer (batch_first=False)
        out = self.transformer_encoder(emb, src_key_padding_mask=src_key_padding_mask)  
        # out shape: [seq_len, batch_size, d_model]
        out = out.transpose(0, 1)  # -> [batch_size, seq_len, d_model]
        return out

# ---------------------
# Speaker & Listener Modules
# ---------------------
class SpeakerNet(nn.Module):
    """Takes an observation token plus existing comm tokens, outputs next comm token."""
    def __init__(self, obs_size, vocab_size, d_model=64, num_heads=4, ff_hidden=128, dropout=0.1):
        super(SpeakerNet, self).__init__()
        # We can define an "effective" vocab = [ real comm vocab + possible obs inputs ]
        self.effective_vocab_size = vocab_size + obs_size
        self.d_model = d_model
        
        self.transformer = TransformerEncoder(
            self.effective_vocab_size, d_model, num_heads, ff_hidden, dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, vocab_size)
        self.critic = nn.Linear(128, 1)

    def forward(self, x, pad_mask=None):
        """
        x shape: [batch_size, seq_len] 
            where each token can be either a "comm token" in [0..vocab_size-1]
            or an "obs token" in [vocab_size..vocab_size+obs_size-1].
        pad_mask shape: [batch_size, seq_len] of booleans (True means "pad" / ignore).
        """
        # Convert pad_mask to shape [batch_size, seq_len] -> pass to transform as [batch_size, seq_len]
        # But the Transformer expects src_key_padding_mask: [batch_size, seq_len].
        out = self.transformer(x, src_key_padding_mask=pad_mask)  # [batch_size, seq_len, d_model]
        # Pool over seq_len dimension (simple mean)
        pooled = out[:, -1, :]
        hidden = self.fc(pooled)
        logits = self.actor(hidden)
        value = self.critic(hidden)
        return logits, value

class ListenerNet(nn.Module):
    """Takes the sequence of comm tokens, outputs a guess about the target obs."""
    def __init__(self, vocab_size, obs_size, d_model=64, num_heads=4, ff_hidden=128, dropout=0.1):
        super(ListenerNet, self).__init__()
        self.d_model = d_model
        self.transformer = TransformerEncoder(
            vocab_size, d_model, num_heads, ff_hidden, dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, obs_size)
        self.critic = nn.Linear(128, 1)

    def forward(self, x, pad_mask=None):
        """
        x shape: [batch_size, seq_len] of comm tokens in [0..vocab_size-1].
        pad_mask shape: [batch_size, seq_len].
        """
        out = self.transformer(x, src_key_padding_mask=pad_mask)
        pooled = out[:, -1, :]
        hidden = self.fc(pooled)
        logits = self.actor(hidden)
        value = self.critic(hidden)
        return logits, value

# ---------------------
# Single Model that holds Speaker & Listener
# ---------------------
class SpeakerListenerModel(nn.Module):
    def __init__(self, obs_size, vocab_size):
        super(SpeakerListenerModel, self).__init__()
        self.speaker_net = SpeakerNet(obs_size, vocab_size)
        self.listener_net = ListenerNet(vocab_size, obs_size)

    def forward_speaker(self, x, pad_mask=None):
        return self.speaker_net(x, pad_mask)

    def forward_listener(self, x, pad_mask=None):
        return self.listener_net(x, pad_mask)