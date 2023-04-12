import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the RNN to classify the sentiment of a sentence with torch and train on GPU
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        for p in self.embedding.parameters():
            p.requires_grad = False
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [sent len, batch size]
        embedded = self.dropout(self.embedding(text))
        # embedded = [sent len, batch size, emb dim]
        output, (hidden, cell) = self.rnn(embedded)
        # output = [sent len, batch size, hid dim * num directions]
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]
        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout
        # hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        hidden = self.dropout(output[-1,:,:])
        # hidden = [batch size, hid dim * num directions]
        return self.fc(hidden)
    
def main():
    # hyperparameters
    learning_rate = 0.0001
    num_epochs = 50
    batch_size = 512
    lbd = 0.0001
    dropout = 0.1
    n_layers = 2

    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    OUTPUT_DIM = 5
    BIDIRECTIONAL = True
    
    ################################
    # DataLoader
    ################################

    # set up fields
    TEXT = data.Field()
    LABEL = data.Field(sequential=False,dtype=torch.long)

    # make splits for data
    # DO NOT MODIFY: fine_grained=True, train_subtrees=False
    train, val, test = datasets.SST.splits(
        TEXT, LABEL, fine_grained=True, train_subtrees=False)

    # print information about the data
    print('train.fields', train.fields)
    print('len(train)', len(train))
    print('vars(train[0])', vars(train[0]))

    # build the vocabulary
    # you can use other pretrained vectors, refer to https://github.com/pytorch/text/blob/master/torchtext/vocab.py
    TEXT.build_vocab(train, vectors=Vectors(name='data/vector.txt', cache='./data'))
    LABEL.build_vocab(train)
    # We can also see the vocabulary directly using either the stoi (string to int) or itos (int to string) method.
    print(TEXT.vocab.itos[:10])
    print(LABEL.vocab.stoi)
    print(TEXT.vocab.freqs.most_common(20))

    # print vocab information
    print('len(TEXT.vocab)', len(TEXT.vocab))
    print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

    # make iterator for splits
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_size=batch_size)

    # print batch information
    batch = next(iter(train_iter)) # for batch in train_iter
    print(batch.text) # input sequence
    print(batch.label) # groud truth

    # Attention: batch.label in the range [1,5] not [0,4] !!!
    # initialize the model
    INPUT_DIM = len(TEXT.vocab)

    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, n_layers, BIDIRECTIONAL, dropout).to(device)

    ###########################################################
    # pre trained word embedding
    ###########################################################
    pretrained_embeddings = TEXT.vocab.vectors
    print(pretrained_embeddings)
    # you should maintain a nn.embedding layer in your network
    model.embedding.weight.data.copy_(pretrained_embeddings)


    # initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lbd)

    # initialize the loss function
    criterion = nn.CrossEntropyLoss().to(device)

    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []
    # train the model

    # print("\n\n\n\n\n\n")
    # for batch in train_iter:
    #     input = batch.text.to(device)
    #     ret = model.rnn(model.dropout(model.embedding(input)))
    #     print(ret)
    #     break
    # exit(0)
    for epoch in range(num_epochs):
        t_loss = 0
        t_accu = 0
        model.train()
        for batch in train_iter:
            optimizer.zero_grad()
            text = batch.text.to(device)
            target = batch.label.to(device)
            predictions = model(text)
            t_loss = criterion(predictions, target-1)
            t_loss.backward()
            optimizer.step()

        t_accu = (predictions.argmax(1) == target-1).sum().item() / target.shape[0]
        training_loss.append(t_loss.item())
        training_accuracy.append(t_accu)
        print(f'| Epoch: {epoch+1:02} | Loss: {t_loss:.3f} | Acc: {t_accu*100:.2f}% |')

        v_loss = 0
        v_accu = 0
        model.eval()
        with torch.no_grad():
            for batch in val_iter:
                text = batch.text.to(device)
                target = batch.label.to(device)
                predictions = model(text).squeeze(1)
                v_loss += criterion(predictions, target-1).item()
                v_accu += (predictions.argmax(1) == target-1).sum().item()

        v_loss /= len(val_iter)
        v_accu /= len(val)
        validation_loss.append(v_loss)
        validation_accuracy.append(v_accu)
        print(f'| Val Loss: {v_loss:.3f} | Val Acc: {v_accu*100:.2f}% |')

    # test the model
    t_loss = 0
    t_accu = 0
    test_loss = 0
    v_accu = 0
    model.eval()
    with torch.no_grad():
        for batch in test_iter:
            text = batch.text.to(device)
            target = batch.label.to(device)
            predictions = model(text).squeeze(1)
            test_loss += criterion(predictions, target-1).item()
            v_accu += (predictions.argmax(1) == target-1).sum().item()

    test_loss /= len(test_iter)
    v_accu /= len(test)
    print(f'| Test Loss: {test_loss:.3f} | Test Acc: {v_accu*100:.2f}% |')

    # plot the loss and accuracy curve
    import matplotlib.pyplot as plt
    plt.plot(training_loss)
    plt.savefig('training_loss.png')
    plt.figure()
    plt.plot(training_accuracy)
    plt.savefig('training_accuracy.png')

    plt.figure()
    plt.plot(validation_loss)
    plt.savefig('validation_loss.png')
    plt.figure()
    plt.plot(validation_accuracy)
    plt.savefig('validation_accuracy.png')

    # save the model
    torch.save(model.state_dict(), 'model.pt')

if __name__ == '__main__':
    main()