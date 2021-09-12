from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk import word_tokenize
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
from utils.tools import EarlyStopping, train_epoch, val_epoch, prediction, texts_to_sequences, pad_features, sent2vec, evaluation, loss_plot, confusion_matrix_plot
from utils.models import Classifier_LSTM
import matplotlib.pyplot as plt
import seaborn as sns


#loading the data from the files
folder = "data/"

xtest, xtrain, ytrain = [], [], []

with open(os.path.join(folder, "xtrain_obfuscated.txt"), "r") as file_content:
    xtrain = (file_content.read().splitlines())

with open(os.path.join(folder, "xtest_obfuscated.txt"), "r") as file_content:
    xtest = (file_content.read().splitlines())

with open(os.path.join(folder, "ytrain.txt"), "r") as file_content:
    ytrain = (file_content.read().splitlines())

# One of the most important task is to visualize data before starting with any ML task.
for i in range(5):
    print(xtrain[i] + "\t: " + ytrain[i][:100] + "...")


# We have used glove vectors as embeddings
embeddings_index = {}
f = open('pretrained_embeddings/glove.6B.300d.txt')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# create sentence vectors using the above function for training and validation set
xtrain_emdd = [sent2vec(x, embeddings_index) for x in (xtrain)]
xtest_emdd = [sent2vec(x, embeddings_index) for x in (xtest)]

xtrain_emdd = np.array(xtrain_emdd)
xtest_emdd = np.array(xtest_emdd)


# Since our dictionary is containing character wise operation. we can create our own dictionary and use it.
vocab_dict = {'a': 11, 'c': 23, 'b': 26, 'e': 5, 'd': 19, 'g': 17,
              'f': 20, 'i': 9, 'h': 2, 'k': 12, 'j': 25, 'm': 3, 'l': 6,
              'o': 24, 'n': 14, 'q': 13, 'p': 10, 's': 15, 'r': 16, 'u': 1,
              't': 8, 'w': 7, 'v': 4, 'y': 21, 'x': 22, 'z': 18}


xtrain_seq = texts_to_sequences(xtrain, vocab_dict)
xtest_seq = texts_to_sequences(xtest, vocab_dict)


max_len = 450
# zero pad the sequences
xtrain_pad = pad_features(xtrain_seq, seq_length=max_len)
xtest_pad = pad_features(xtest_seq, seq_length=max_len)


# create an embedding matrix for the words we have in the dataset
embedding_matrix = np.zeros((len(vocab_dict) + 1, 300))
for word, i in tqdm(vocab_dict.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

ytrain_enc = np.array(list(map(int, ytrain)))


# split the data into test, validation and train

x_train, x_test, y_train, y_test = train_test_split(
    xtrain_pad, ytrain_enc, train_size=0.7, random_state=1)

x_test, x_val, y_test, y_val = train_test_split(
    x_test, y_test, test_size=0.5, random_state=1)

print("x_train", x_train.shape)
print("y_train", y_train.shape)
print("x_val", x_val.shape)
print("y_val", y_val.shape)
print("x_test", x_test.shape)
print("y_test", y_test.shape)


# create Tensor datasets for train, test and val
train_data = TensorDataset(torch.from_numpy(
    x_train), torch.from_numpy(y_train))
valid_data = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
test_data = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

# Tensor dataset used for final prediction
final_test_data = TensorDataset(torch.from_numpy(xtest_pad))

# dataloaders
batch_size = 512

# make sure to SHUFFLE your training data. Keep Shuffle=True.
train_loader = DataLoader(train_data, shuffle=True,
                          batch_size=batch_size, drop_last=True)
valid_loader = DataLoader(valid_data, shuffle=True,
                          batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_data, shuffle=True,
                         batch_size=batch_size, drop_last=True)
# Data loader used for final prediction
final_test_loader = DataLoader(
    final_test_data, shuffle=True, batch_size=batch_size, drop_last=True)


# obtain one batch of training data and label.
dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

print('Sample input size: ', sample_x.size())  # batch_size, seq_length
print('Sample input: \n', sample_x)
print()
print('Sample label size: ', sample_y.size())  # batch_size
print('Sample label: \n', sample_y)


# Check if GPU is available.
train_on_gpu = torch.cuda.is_available()

if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')

# loss and optimization functions
# Instantiate the model with these hyperparameters
vocab_size = 27  # +1 for the 0 padding + our word tokens
output_size = 12
embedding_dim = 300
hidden_dim = 256
n_layers = 2
epochs = 100
lr = 0.001
clip = 5
criterion = nn.CrossEntropyLoss()

model = Classifier_LSTM(vocab_size, output_size, embedding_dim,
                        hidden_dim, n_layers, embedding_matrix, train_on_gpu)


optimizer = torch.optim.Adam(model.parameters(), lr=lr)
print(model)


# Instantiate early stopping class to avoid overfit provide the location for saving checkpoint.
early_stopping = EarlyStopping(
    patience=10, verbose=True, path="checkpoints/model.pt")


# move model to GPU, if available
if(train_on_gpu):
    model.cuda()

# Main loop alternatingly calling the train and valid epoch and checking for overfit
total_train_loss, total_val_loss = [], []
for i in range(epochs):
    train_loss = train_epoch(
        model, train_loader, optimizer, clip, criterion, batch_size, train_on_gpu)
    val_loss = val_epoch(model, valid_loader, criterion,
                         batch_size, train_on_gpu)
    total_train_loss.append(train_loss)
    total_val_loss.append(val_loss)
    print("epoch ", i, " train_loss ", train_loss, "val_loss ", val_loss)
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

# Loading the optimal model which does not overfit and performs best.
model.load_state_dict(torch.load(
    'checkpoints/model.pt'))


loss_plot(total_train_loss,total_val_loss)


# Evaluation of the test data which is splitted from train to get how the model is performing.

all_predictions,all_labels = evaluation(model,batch_size,test_loader,train_on_gpu,criterion)

# Data provided in the instructions.
dictionary = {0: "alice_in_wonderland",
              1: "dracula",
              2: "dubliners",
              3: "great_expectations",
              4: "hard_timesN",
              5: "huckleberry_finn",
              6: "les_miserable",
              7: "moby_dick",
              8: "oliver_twist",
              9: "peter_pan",
              10: "tale_of_two_cities",
              11: "tom_sawyer"}


def dict_returns(num):
    return dictionary[num]


all_predictions = (map(dict_returns, all_predictions))
all_labels = (map(dict_returns, all_labels))
all_predictions = list(all_predictions)
all_labels = list(all_labels)


# This report will provide recall/precision/F1 score along with weighted accuracy for each of the classes
print('Classification Report:')
print(classification_report(all_predictions, all_labels,
                            labels=(list(set(all_labels))), digits=3))

#This function will save the plot of confusion matrix in the output folder
confusion_matrix_plot(all_predictions,all_labels)

# Final method to save a text file for the test file given.
prediction(model, batch_size, train_on_gpu, final_test_loader)
