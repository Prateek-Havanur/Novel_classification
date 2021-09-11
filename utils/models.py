import torch.nn as nn
import torch


def create_emb_layer(weights_matrix, non_trainable=False):
    """Creates and loads the weight matrix created by us from GLove embedding"""

    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.weight.data.copy_(torch.from_numpy(weights_matrix).cuda())
    if non_trainable:
        # let the weights be fixed. No need to update it during back prop
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

class Classifier_LSTM(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers,embedding_matrix,train_on_gpu, drop_prob=0.5,):
        """
         We are training the embedded layers along with LSTM for the sentiment analysis
        """
        super(Classifier_LSTM, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.train_on_gpu = train_on_gpu

        # embedding layer and LSTM layers
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(
            embedding_matrix, True)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)

        # dropout layer to avoid overfitting
        self.dropout = nn.Dropout(0.5)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)

#         reshape to be batch_size first
        out = out.view(batch_size, -1, self.output_size)
        out = out[:, -1, :]  # get last batch of labels

        # return last sigmoid output and hidden state
        return out, hidden

    def init_hidden(self, batch_size):
        # initilizing hidden layers
        weight = next(self.parameters()).data

        if (self.train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden