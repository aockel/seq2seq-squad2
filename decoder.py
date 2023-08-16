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
        self.lstm = nn.LSTM(embedding_dim, self.hidden_size, num_layers=self.lstm_layer, batch_first=True, dropout=dropout)
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
        # embeds the input token using an embedding layer of size (input_size, hidden_size)
        decoder_input = decoder_input.unsqueeze(1)  # add a time dimension of size 1
        embedded = self.dropout(self.embedding(decoder_input))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.lin_out(output.squeeze(1))  # remove the time dimension
        prediction = self.softmax(prediction)
        return prediction, hidden, cell
