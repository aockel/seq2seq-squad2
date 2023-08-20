import torch.nn as nn


class Decoder(nn.Module):
    """
        Inputs:
            input_size: size of the input vocabulary
            hidden_size: of the LSTM
            output_size: size of the output vocabulary
            embedding_size: embedding dimension
            lstm_layer: number of lstm layers Default = 1
            dropout: Default = 0.5
    """

    def __init__(self, input_size, hidden_size, output_size, embedding_dim, lstm_layer=1, dropout=0.5):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm_layer = lstm_layer

        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_size, num_layers=self.lstm_layer, dropout=dropout)
        # The LSTM produces an output by passing the hidden state to the Linear layer
        self.lin_out = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(dropout)
        # optional test with LogSoftmax
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, decoder_input, hidden, cell):
        """
            Inputs:
                decoder_input: previous token in the sequence
                hidden: previous hidden the state
                cell: previous cell state
            Outputs:
                prediction: the prediction
                hidden: the hidden state
                cell: the cell state
        """
        # decoder_input shape is torch.Size([batch_size])
        # add a time dimension of size 1
        decoder_input = decoder_input.unsqueeze(0)
        # torch.Size([1, batch_size])
        # embeds the input token using an embedding layer of size (input_size, hidden_size)
        embedded = self.dropout(self.embedding(decoder_input))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # torch.Size([1, batch_size, hidden_size])
        # prediction = self.lin_out(output.squeeze(1))  # remove the time dimension
        output = self.lin_out(output.squeeze(0))  # remove the time dimension
        # torch.Size([batch_size, vocab_size])
        output = self.softmax(output)
        # output shape torch.Size([batch_size, vocab_size])
        return output, hidden, cell
