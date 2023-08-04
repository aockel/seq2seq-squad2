import torch
import torch.nn as nn
import torch.optim as optim
# from torchtext.legacy.data import Field, BucketIterator


from encoder import Encoder
from decoder import Decoder
from seq2seq_old import Seq2Seq
from vocab import Vocab
from reader import read_text
from helpers import train, evaluate

df = read_text()
vocab = Vocab(name='firstVocab')

input_size = len(vocab.words)
output_size = 15
embedding_size = 256
hidden_size = 512
lstm_layer = 2
dropout = 0.5
epochs = 10
clip = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# BATCH_SIZE = 128
# train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
#    (train_data, valid_data, test_data),
#    batch_size = BATCH_SIZE,
#    device = device)

# instantiate encoder and decoder classes
enc = Encoder(input_size, hidden_size, embedding_size, lstm_layer, dropout)
dec = Decoder(hidden_size, output_size, embedding_size, lstm_layer, dropout)
# instantiate seq2seq model
model = Seq2Seq(enc, dec, device).to(device)
# definer optimizer and loss function
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

best_valid_loss = float('inf')

for epoch in range(epochs):
    # model, iterator, optimizer, criterion, clip
    train_loss = train(model, df, optimizer, criterion, clip)
    valid_loss = evaluate(model, df, criterion)


    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')

    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {train_loss:7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {valid_loss:7.3f}')

model.load_state_dict(torch.load('tut1-model.pt'))

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')