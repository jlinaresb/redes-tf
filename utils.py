from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def get_mlp_model(columns, hiddenLayerOne, hiddenLayerTwo,
	dropout, learnRate):
	# initialize a sequential model
	# input data
	model = Sequential()

	model.add(Dense(columns, activation="relu", input_dim=columns))
	model.add(Dense(hiddenLayerOne, activation="relu"))
	model.add(Dropout(dropout))
	model.add(Dense(hiddenLayerTwo, activation="relu"))
	model.add(Dense(1))

	# compile the model
	model.compile(
		optimizer=Adam(learning_rate=learnRate),
		loss="mean_squared_logarithmic_error",
		metrics=['RootMeanSquaredError','mse', 'mae', 'mape', 'cosine_proximity'])
	# return compiled model
	return model