import torch
import torch.nn as nn


class Encoder(torch.nn.Module):
    """
        Inputs:
            input_size: size of the vocabulary
            hidden_size: of the LSTM
            embedding_dim: embedding dimension
            lstm_layer: Default = 1
            dropout: Default = 0.5
    """

    def __init__(self, input_size, hidden_size, embedding_dim, lstm_layer=1, dropout=0.5, batch_size=1):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.lstm_layer = lstm_layer
        self.batch_size = batch_size

        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_size, num_layers=self.lstm_layer, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_input):
        """
            Inputs:
                encoder_input: the src vector
            Outputs:
                hidden: the hidden state
                cell: the cell state
        """
        # embeds the input sequence using an embedding layer of size (input_size, embedding_dim),
        embed = self.dropout(self.embedding(encoder_input))
        outputs, (hidden, cell) = self.lstm(embed)
        return hidden, cell
