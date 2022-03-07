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

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV

# Function to create the model
def get_mlp_model(hiddenLayerOne, hiddenLayerTwo, learnRate):
	# initialize a sequential model
	# input data
	model = Sequential()
	model.add(Dense(10, activation="relu", input_dim=10))
	model.add(Dense(hiddenLayerOne, activation="relu"))
	model.add(Dense(hiddenLayerTwo, activation="relu"))
	model.add(Dense(1))
	# compile the model
	model.compile(
		optimizer=Adam(learning_rate=learnRate),
		loss="mean_squared_logarithmic_error",
		metrics=['RootMeanSquaredError','mse', 'mae', 'mape', 'cosine_proximity'])
	# return compiled model
	return model


# Arguments
inputDir = '/mnt/netapp2/Store_uni/home/ulc/co/jlb/redes-tf/data/'
outDir = '/mnt/netapp2/Store_uni/home/ulc/co/jlb/redes-tf/models/'
#filename = 'example.csv'
outfile = filename.replace('.csv', '')

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("-f","--filename", help="Filename of input data",
                    type=str, required=True)
args = parser.parse_args()
filename = args.filename



# working directory to input data
os.chdir(inputDir)

# load dataset
data = pd.read_csv(filename, index_col = 0)

# Train/test split
X = data.drop('target', axis = 1)
y = data.target

X = np.asarray(X).astype(np.float32)
y =  np.asarray(y).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(len(X_train), 'train examples')
print(len(X_test), 'test examples')

# wrap our model into a scikit-learn compatible classifier
print("[INFO] initializing model...")
model = KerasRegressor(build_fn=get_mlp_model, verbose=0)

# define a grid of the hyperparameter search space
hiddenLayerOne = [16, 4]
hiddenLayerTwo = [4, 3]
learnRate = [1e-2, 1e-1]
batchSize = [16, 10]
epochs = [5, 6]

# create a dictionary from the hyperparameter grid
grid = dict(
	hiddenLayerOne=hiddenLayerOne,
	learnRate=learnRate,
	hiddenLayerTwo=hiddenLayerTwo,
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
searcher = RandomizedSearchCV(estimator=model, n_jobs=-1, cv=3,
	param_distributions=grid, scoring="neg_mean_squared_error", verbose = 20)
searchResults = searcher.fit(X_train, y_train)

# summarize grid search information
bestScore = searchResults.best_score_
bestParams = searchResults.best_params_
print("[INFO] best score is {:.2f} using {}".format(bestScore,
	bestParams))

# extract the best model, make predictions on our data, and show a
# model report
print("[INFO] evaluating the best model...")
bestModel = searchResults.best_estimator_
performance = bestModel.score(X_test, y_test)
print("Mean squared error: {:.2f}".format(performance))


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