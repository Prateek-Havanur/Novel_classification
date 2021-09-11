import torch
import torch.nn as nn
import numpy as np

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if True:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        print(f'Saved the model successfully')

def train_epoch(model, train_loader, optimizer, clip, criterion,batch_size,train_on_gpu):
    """Train loop. iterates batches in from trainloader and calculates and returns ephoch loss."""
    h = model.init_hidden(batch_size)
    epoch_loss = 0
    # batch loop
    model.train()
    for inputs, labels in train_loader:
        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])
        # zero accumulated gradients
        optimizer.zero_grad()
        # get the output from the model
        output, h = model(inputs, h)
        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.long())
        epoch_loss = epoch_loss + loss
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    epoch_loss = epoch_loss.cpu().detach().numpy()
    return epoch_loss

def val_epoch(model, valid_loader, criterion,batch_size,train_on_gpu):
    """Train loop. iterates batches in from validiloader and calculates and returns ephoch loss."""
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        val_h = model.init_hidden(batch_size)
        val_losses = []
        model.eval()
        for inputs, labels in valid_loader:
            val_h = tuple([each.data for each in val_h])
            if(train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()
            output, val_h = model(inputs, val_h)
            val_loss = criterion(output.squeeze(), labels.long())
            val_losses.append(val_loss.item())
            epoch_loss = epoch_loss + val_loss
    epoch_loss = epoch_loss.cpu().detach().numpy()
    return epoch_loss


all_final_predictions = []

def prediction(model,batch_size,train_on_gpu,final_test_loader):
    """iterates through the test_loader and creates file containing prediction similiar to y_train given in the task."""
    h = model.init_hidden(batch_size)
    model.cuda()
    model.eval()
    # iterate over test data
    for inputs in final_test_loader:
        h = tuple([each.data for each in h])
        inputs = torch.from_numpy(np.array(inputs[0]))
        if(train_on_gpu):
            inputs = inputs.cuda()
        # get predicted outputs
        output, h = model(inputs, h)
        # rounds to the nearest integer
        pred = torch.argmax(output.squeeze(), dim=1)
        all_final_predictions.extend(pred.cpu().detach().numpy())
        with open("../outputs/ytest.txt", "w") as txt_file:
            for line in all_final_predictions:
                txt_file.write(str(line) + "\n")

def texts_to_sequences(data,vocab_dict):
    sequence = []
    for sentence in data:
        sentence_seq = []
        for word in sentence:
            sentence_seq.append(vocab_dict[word])
        sequence.append(sentence_seq)
    return sequence


# Returns fixed length feature vector.Based on the lengthit either truncated or padds with leading zeros.
def pad_features(seq_ints, seq_length):
    """ Return features of seq_ints, where each sequence is padded with 0's 
        or truncated to the input seq_ints.
    """
    # getting the correct rows x cols shape
    features = np.zeros((len(seq_ints), seq_length), dtype=int)

    # for each review, I grab that review and
    for i, row in enumerate(seq_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features

def tokenize_chars(s):
    return list(s)


def sent2vec(s,embeddings_index):
    """ it will create a normalised vector for each sentence using embedding index provided.
    """
    words = str(s).lower().encode().decode('utf-8')
    words = tokenize_chars(words)
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(450)
    return v / np.sqrt((v ** 2).sum())