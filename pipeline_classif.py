import os
import pandas as pd
import numpy as np
import json
import argparse

from sklearn.model_selection import train_test_split

# import the necessary packages
import tensorflow as tf
tf.random.set_seed(1993)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV



# Function to create the model
def get_mlp_model(firstLayer, hiddenLayerOne, hiddenLayerTwo, dropout, learnRate):
	# initialize a sequential model
	# input data
	model = Sequential()
	model.add(Dense(firstLayer, activation="relu", input_dim=10))
	model.add(Dense(hiddenLayerOne, activation="relu"))
	model.add(Dropout(dropout))
	model.add(Dense(hiddenLayerTwo, activation="relu"))
	model.add(Dropout(dropout))
	model.add(Dense(3), activation = "softmax")
	# compile the model
	model.compile(
		optimizer=Adam(learning_rate=learnRate),
		loss="sparse_categorical_crossentropy",
		metrics=['accuracy'])
	# return compiled model
	return model


# Directories
inputDir = '/mnt/netapp2/Store_uni/home/ulc/co/jlb/redes-tf/data/'
outDir = '/mnt/netapp2/Store_uni/home/ulc/co/jlb/redes-tf/models/'

outfile = 'rituximab'

# load datasets (train/test)
train = pd.read_csv(inputDir + 'data_rituximab_train.csv', index_col = 0)
test = pd.read_csv(inputDir + 'data_rituximab_test.csv', index_col = 0)

# convert variable response to int
train['target'].replace(['non_responder', 'moderate_responder', 'responder'], [0, 1, 2], inplace=True)
test['target'].replace(['non_responder', 'moderate_responder', 'responder'], [0, 1, 2], inplace=True)

X_train = train.drop('target', axis = 1)
y_train = train.target
y_train = np.asarray(y_train).astype(np.float32)

X_test = test.drop('target', axis = 1)
y_test = test.target
y_test = np.asarray(y_test).astype(np.float32)


# Train/test split
nInputlayer = len(X_train.columns)
X_train = np.asarray(X_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)

print(len(X_train), 'train examples')
print(len(X_test), 'test examples')

# wrap our model into a scikit-learn compatible classifier
print("[INFO] initializing model...")
model = KerasClassifier(build_fn=get_mlp_model, verbose=0)

# define a grid of the hyperparameter search space
hiddenLayerOne = [32, 16, 8, 4]
hiddenLayerTwo = [5, 4, 3, 2]
learnRate = [1e-2, 1e-3, 1e-4]
dropout = [0.3, 0.4, 0.5]
batchSize = [4, 8, 16, 32]
epochs = [20, 40, 80, 160, 500]

# create a dictionary from the hyperparameter grid
grid = dict(
	firstLayer=[nInputlayer],
	hiddenLayerOne=hiddenLayerOne,
	learnRate=learnRate,
	hiddenLayerTwo=hiddenLayerTwo,
	dropout=dropout,
	batch_size=batchSize,
	epochs=epochs
)

# initialize a random search with a 3-fold cross-validation and then
# start the hyperparameter search process
'''
Metrics:
https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
'''
print("[INFO] performing random search...")
searcher = GridSearchCV(estimator=model, n_jobs=-1, cv=3,
	param_grid=grid, scoring="accuracy", verbose = 20)
searchResults = searcher.fit(X_train, y_train)

# summarize grid search information
bestScore = searchResults.best_score_
bestParams = searchResults.best_params_
print("[INFO] best score is {:.2f} using {}".format(bestScore, bestParams))

# extract the best model, make predictions on our data, and show a
# model report
print("[INFO] evaluating the best model...")
bestModel = searchResults.best_estimator_
accuracy = bestModel.score(X_test, y_test)
print("accuracy: {:.2f}%".format(accuracy * 100))


# Save results
if not os.path.exists(outDir + outfile):
    os.makedirs(outDir + outfile)

# serialize model to JSON
model_json = bestModel.model.to_json()
with open(outDir + outfile + "/model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
bestModel.model.save(outDir + outfile + "/model.h5")
print("Saved model to disk")

os.makedirs(outDir + outfile + '/test')
np.savez(outDir + outfile + 'test' + '/test.npz', X_test=X_test, y_test=y_test)