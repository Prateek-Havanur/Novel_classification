# Novel classification 

**Tech stack**
- PyTorch
- LSTM
- Python

**Folder structure details**
- novel_classification.py is main file running the entire project.
- util/tool.py contains the functions like early stopping/ train epoch/ test epoch etc.
- output folder contains the images of confusion matrix and training plot along with prediction file.
- logs folder contains logs of the running the code.
- checkpoints contains the model saved.
- data contains input files and embedding file which contains glove embeddings.
- Models folder contains LSTM model used.

This project solves the task of training a Machine/Deep Learning model that classifies a given line as belonging to one of the following 12 novels:

- 0: alice_in_wonderland
- 1: dracula
- 2: dubliners
- 3: great_expectations
- 4: hard_timesN
- 5: huckleberry_finn
- 6: les_miserable
- 7: moby_dick
- 8: oliver_twist
- 9: peter_pan
- 10: tale_of_two_cities
- 11: tom_sawyer.

**Data provided:**

Input data contains the obsfuscated text which will avoid some of the traditional tasks like removing stop word, lemmatization, etc

**Approaches tried:**

Since the the data provided was in obfuscated format i thought transformers wouldnt be the best choice since they contain preatrained tokenizer which are by default subword tokenizers and character level tokenizers are needed for the given dataset. In order to confirm my assumption i tried the XLNet and got accuracy of 36%.   LSTM was the approach finalised and explained below.

|              | LSTM  | XLNet Transformers |
|--------------|--------------|--------------|
| Accuracy     | 69%     | 36%     |

**LSTM:**

The file novel_classfication.py is the file which contains the code to train,test and generate the prediction file as mentioned in the task description.
I have used pretrained glove embeddings to intialiase weight of embedding layer. Data has been split into train/Validation/test with 70/15/15 ratios respectively.

Below is the structure of the model

Classifier_LSTM(
  (embedding): Embedding(27, 300)
  (lstm): LSTM(300, 256, num_layers=2, batch_first=True, dropout=0.5)
  (dropout): Dropout(p=0.5, inplace=False)
  (fc): Linear(in_features=256, out_features=12, bias=True)
  (sig): Sigmoid()
)

Early stopping has been used to avoid overfit in the training process. The accuracy of the model is 69% accuracy on test set. Below is the image of confusion matrix classwise. 


![alt text](https://raw.githubusercontent.com/Prateek-Havanur/Novel_classification/main/outputs/confusion%20matrix.png)

**Improvements**
- We can try to experiment the task with different models like RNN/Bidirectional RNN/LSTM, GRU etc.
- We can also create a simple jupyter notebook explaining details about printing sample data and visualization classwise data etc.
- The code and be more modulised and improved by using argparse for and using arguments to run code example learning rate/batch size etc.
