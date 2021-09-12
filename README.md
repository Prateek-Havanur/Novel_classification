# Novel classification 

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

**Tech stack**
- PyTorch
- LSTM
- Python

**Folder structure details**
- novel_classification.py is main file running the entire project.
- util/tool.py contains the functions like early stopping/ train epoch/ test epoch etc.
- output folder contains the images of confusion matrix and training plot along with prediction file. This folder also contains log file showing details while code   runs and all the results including classification report
- checkpoints contains the model saved.
- data contains input files and embedding file which contains glove embeddings.
- Models folder contains LSTM model used.



**Data provided:**

Input data contains the obsfuscated text which will avoid some of the traditional tasks like removing stop word, lemmatization, etc

**How to run**
- Clone the repository
- pip install -r /path/to/requirements.txt
- Place the input data in data folder and emeddings in pretrained_emeddings folder
- Execute python novel_classfication.py in terminal.

**Approaches tried:**

Since the the data provided was in obfuscated format transformers wouldnt be the best choice since they contain preatrained tokenizer which are by default subword tokenizers and character level tokenizers are needed for the given dataset. In order to confirm the assumption XLNet was trained and tested to get accuracy of 36%.  LSTM was the approach finalised and explained below.

|              | LSTM  | XLNet Transformers |
|--------------|--------------|--------------|
| Accuracy     | 69%     | 36%     |

**LSTM:**

The file novel_classfication.py is the file which contains the code to train,test and generate the prediction file as mentioned in the task description.
I have used pretrained glove embeddings to intialiase weight of embedding layer. Data has been split into train/Validation/test with 70/15/15 ratios respectively.

Below is the structure of the model

Classifier_LSTM(\
  (embedding): Embedding(27, 300)\
  (lstm): LSTM(300, 256, num_layers=2, batch_first=True, dropout=0.5)\
  (dropout): Dropout(p=0.5, inplace=False)\
  (fc): Linear(in_features=256, out_features=12, bias=True)\
  (sig): Sigmoid()\
)

Early stopping has been used to avoid overfit in the training process. The accuracy of the model is 69% accuracy on test set. Below is the image of confusion matrix classwise and training process. 



<p align="center">
  <img src="https://raw.githubusercontent.com/Prateek-Havanur/Novel_classification/main/outputs/confusion%20matrix.png" width="350" title="hover text">
  <img src="https://raw.githubusercontent.com/Prateek-Havanur/Novel_classification/main/outputs/Loss_plot_plain_model.png" width="350" alt="accessibility text">
</p>

**Classification report**

|                    | precision    |recall  |f1-score   |support|
|--------------------|--------------|--------|-----------|-------| 
|      les_miserable |     0.711    | 0.746  |   0.728   |    555|
|   huckleberry_finn |     0.866    | 0.929  |   0.896   |    326|
|            dracula |     0.753    | 0.645  |   0.695   |    575|
|       oliver_twist |     0.846    | 0.836  |   0.841   |    506|
|          dubliners |     0.646    | 0.703  |   0.673   |    195|
| tale_of_two_cities |     0.636    | 0.518  |   0.571   |    515|
| great_expectations |     0.725    | 0.728  |   0.726   |    578|
|          peter_pan |     0.514    | 0.449  |   0.480   |    158|
|          moby_dick |     0.781    | 0.833  |   0.807   |    678|
|alice_in_wonderland |     0.630    | 0.793  |   0.702   |     58|
|        hard_timesN |     0.761    | 0.819  |   0.789   |    304|
|         tom_sawyer |     0.507    | 0.662  |   0.575   |    160|
|           accuracy |              |        |   0.732   |   4608|
|          macro avg |     0.698    | 0.722  |   0.707   |   4608|
|       weighted avg |     0.731    | 0.732  |   0.730   |   4608|

**Improvements**
- We can try to experiment the task with different models like RNN/Bidirectional RNN/LSTM, GRU etc.
- We can also create a simple jupyter notebook explaining details about printing sample data and visualization classwise data etc.
- The code and be more modulised and improved by using argparse for and using arguments to run code example learning rate/batch size etc.
