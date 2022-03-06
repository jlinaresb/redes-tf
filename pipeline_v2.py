import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# import the necessary packages
import tensorflow as tf
tf.random.set_seed(1993)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

# Function to create the model
def get_mlp_model(hiddenLayerOne = 5, hiddenLayerTwo = 5,
	dropout = 0, learnRate = 0.01):
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


# working directory to input data
os.chdir('/mnt/netapp2/Store_uni/home/ulc/co/jlb/redes-tf/data/')

# load dataset
data = pd.read_csv('example.csv', index_col = 0)

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
model = KerasClassifier(build_fn=get_mlp_model, verbose=0)

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
searcher = RandomizedSearchCV(estimator=model, n_jobs=1, cv=3,
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
print("Mean squared error: {:.2f}%".format(performance))