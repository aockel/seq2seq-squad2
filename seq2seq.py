import torch
import torch.nn as nn
import random


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg=None, teaching=0.5):
        """
        Inputs:
            src: source sequence tensor of shape (batch_size, src_len)
            trg: target sequence tensor of shape (batch_size, trg_len), or None for inference
            teaching: the probability of using teacher forcing
        Outputs:
            outputs: decoder outputs tensor of shape (batch_size, trg_len, trg_vocab_size)
        """
        if trg is not None:
            # use ground truth target tokens for teacher forcing
            batch_size, trg_len = trg.shape
            trg_vocab_size = self.decoder.output_size
            # tensor to store decoder outputs
            outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
            # last hidden state and cell state of the encoder as initial hidden and cell state for the decoder
            hidden, cell = self.encoder(src)
            # use the first target token as the initial decoder input
            inp = trg[:, 0]
            for t in range(1, trg_len):
                output, hidden, cell = self.decoder(inp, hidden, cell)
                outputs[:, t] = output
                if random.random() < teaching:
                    # use ground truth target token
                    inp = trg[:, t]
                else:
                    # use predicted token as input for next timestep
                    inp = output.argmax(dim=1)
            return outputs
        else:
            # inference mode
            batch_size, max_len = src.shape
            trg_vocab_size = self.decoder.output_size
            # tensor to store decoder outputs
            outputs = torch.zeros(batch_size, max_len, trg_vocab_size).to(self.device)
            # last hidden state and cell state of the encoder as initial hidden and cell state for the decoder
            hidden, cell = self.encoder(src)
            # use the <sos> token as the initial decoder input
            inp = torch.ones(batch_size).to(self.device, dtype=torch.long)
            for t in range(1, max_len):
                output, hidden, cell = self.decoder(inp, hidden, cell)
                outputs[:, t] = output
                # use predicted token as input for next timestep
                inp = output.argmax(dim=1)
            return outputs
