import torch
import torch.nn as nn
import random


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg=None, teaching=0.5, max_len: int = 10):
        """
        Inputs:
            src: source sequence tensor of shape (batch_size, src_len)
            trg: target sequence tensor of shape (batch_size, trg_len), or None for inference
            teaching: the probability of using teacher forcing
            max_len: maximum length of output for inference only
        Outputs:
            outputs: decoder outputs tensor of shape (batch_size, trg_vocab_size, trg_len)
        """

        batch_size, trg_length = trg.shape if trg is not None else src.shape
        trg_len = trg_length if trg is not None else max_len

        trg_vocab_size = self.decoder.output_size
        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_vocab_size, trg_len).to(self.device)
        # last hidden state and cell state of the encoder as initial hidden and cell state for the decoder
        hidden, cell = self.encoder(src)
        # use the <SOS> token as the initial decoder input (vocab 0=<SOS>)
        inp = torch.zeros(batch_size).to(self.device, dtype=torch.long)
        # use the first target token as the initial decoder input
        # inp = trg[:, 0]
        for t in range(trg_len):
            output, hidden, cell = self.decoder(inp, hidden, cell)
            # output shape torch.Size([batch_size, vocab_size])
            outputs[:, :, t] = output
            # torch.Size([batch_size, vocab_size, trg_len])
            if random.random() < teaching:
                # use ground truth target token
                inp = trg[:, t]
            else:
                # use predicted token as input for next timestep
                inp = output.argmax(dim=1)
        return outputs
