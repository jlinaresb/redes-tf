import os

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

# working directory to input data
os.chdir('/mnt/netapp2/Store_uni/home/ulc/co/jlb/redes-tf/data/')

# load dataset
data = pd.read_csv('example.csv', index_col = 1)

# check if 'target' variable is present
if 'target' in data.columns:
    pass
else:
    ValueError("DataFrame must contain target variable!")

# check that all variables are float

# check if NA´s
data.isna()
data.isna().any()
data.isna().sum()

# remove NA´s
data = data.dropna(axis = 0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X = data.drop('target', axis = 1)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)
print(len(X_train), 'train examples')
print(len(X_val), 'validation examples')
print(len(X_test), 'test examples')

# X, Y



# HYPERPARAMETER TUNING
# check this: https://analyticsindiamag.com/parameter-tuning-tensorboard/
# ======== 

# layers
'''
Only two layers
'''
HP_NUM_UNITS1 = hp.HParam('num_units 1', hp.Discrete([4,8,16])) 
HP_NUM_UNITS2 = hp.HParam('num_units 2', hp.Discrete([4,8]))

# Dropout percentaje
'''
Remove conexions between neurons
'''
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.2, 0.5))

# Optimizer
'''
Check diferent optimizer functions
SGD: stochastic gradient descent
Adam: popular extension to stochastic gradient descent
RMSprop: ???
'''
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['SGD', 'Adam', 'RMSprop']))

# Learning rate
HP_OPTIMIZER_LR = hp.HParam('learning_rate', hp.Discrete([0.0001, 0.001, 0.1]))

# Regularization
HP_L2 = hp.HParam('l2 regularizer', hp.RealInterval(.001,.01))

# Activation layer
HP_ACTIVATION_LAYER = hp.HParam('activation_layers', hp.Discrete(['relu', 'tanh']))

# Metric to evaluate the model
METRIC_RMSE = 'RootMeanSquaredError'

#Loop para tunear también la learning rate, también vale para tunear más el optimizador según cuál sea, tienes más info de **args y **kwars en TF :)
optimizer_name = hparams[HP_OPTIMIZER]
learning_rate = hparams[HP_OPTIMIZER_LR]
if optimizer_name == "RMSprop":
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
elif optimizer_name == "SGD":
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
elif optimizer_name == "Adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
else:
    raise ValueError("unexpected optimizer name: %r" % (optimizer_name,))

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS1, HP_NUM_UNITS2, HP_DROPOUT, HP_L2, HP_OPTIMIZER, HP_ACTIVATION_LAYER],
    metrics=[hp.Metric(METRIC_RMSE, display_name='RMSE')],
  )

# Define model 
# ======
def train_test_model(hparams):
  model = tf.keras.Sequential([
    layer.Dense(len(train.columns) - 1, input_shape = len(train.columns) - 1, activation=hparams[HP_ACTIVATION_LAYER]),
    layers.Dense(hparams[HP_NUM_UNITS1], kernel_regularizer=tf.keras.regularizers.l2(hparams[HP_L2]), activation=hparams[HP_ACTIVATION_LAYER]),
    layers.Dropout(hparams[HP_DROPOUT]),
    layers.Dense(hparams[HP_NUM_UNITS2], kernel_regularizer=tf.keras.regularizers.l2(hparams[HP_L2]), activation=hparams[HP_ACTIVATION_LAYER]),
    tf.keras.layers.Dense(1)
  ])
  model.compile(optimizer=optimizer,
                loss='mean_squared_logarithmic_error',
                metrics=['RootMeanSquaredError'])
  model.fit(X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=5) #1000
  _, performance = model.evaluate(val_ds)
  return performance

# Log an hparams summary
def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  
    rmse = train_test_model(hparams)
    tf.summary.scalar(METRIC_RMSE, rmse, step=10)

# Run
session_num = 0
for num_units1 in HP_NUM_UNITS1.domain.values:
  for num_units2 in HP_NUM_UNITS2.domain.values:
    for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
      for l2 in (HP_L2.domain.min_value, HP_L2.domain.max_value):
        for optimizer in HP_OPTIMIZER.domain.values:
            for optimizer_lr in HP_OPTIMIZER_LR.domain.values:
                for activation in HP_ACTIVATION_LAYER.domain.values:
                    hparams = {
                        HP_NUM_UNITS1: num_units1,
                        HP_NUM_UNITS2: num_units2,
                        HP_DROPOUT: dropout_rate,
                        HP_L2: l2,
                        HP_OPTIMIZER: optimizer,
                        HP_OPTIMIZER_LR: optimizer_lr,
                        HP_ACTIVATION_LAYER: activation
                    }
                    run_name = "run-%d" % session_num
                    print('--- Starting trial: %s' % run_name)
                    print({h.name: hparams[h] for h in hparams})
                    run('logs/hparam_tuning/' + run_name, hparams)
                    session_num += 1


# See those params
data_items = hparams.items()
data_list = list(data_items)

df = pd.DataFrame(data_list)

# Get best optimizer and tunning learning rate
optimizer_name = df[1][4]
learning_rate = df[1][5]
if optimizer_name == "RMSprop":
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
elif optimizer_name == "SGD":
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
elif optimizer_name == "Adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Make a ANN for regression with best architecture tunned
def ANN_model_tunned(feature_layer= feature_layer, hidden_1=df[1][0], hidden_2=df[1][1], dropout=df[1][2], l2=df[1][3], optimizer=optimizer, activation_layer=df[1][6]):
    """
    Function to create the architecture of model of Dopamine ANN with TF
    :params: features - features of the model in TF format
    :params: hidden_1 - number of neurons in the first hidden layer
    :params: hidden_2 - number of neurons in the second hidden layer
    :params: dropout - percentage of dropout
    :params: l2 - percentage of l2 optimizer
    :params: optimizer - optmizer compiling model
    :params: optimizer_lr - learning rate tunning the optimizer
    :params: activation_layer - activation function for the layers
    :return: model - model of the ANN
    """
    model = tf.keras.Sequential([
        feature_layer,
        layers.Dense(hidden_1, kernel_regularizer=tf.keras.regularizers.l2(l2), activation=activation_layer),
        layers.Dropout(dropout),
        layers.Dense(hidden_2, kernel_regularizer=tf.keras.regularizers.l2(l2), activation=activation_layer),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=optimizer,
                loss='mean_squared_logarithmic_error',
                metrics=['RootMeanSquaredError','mse', 'mae', 'mape', 'cosine_proximity'])
    return model

model = ANN_model_tunned(feature_layer= feature_layer, hidden_1=df[1][0], hidden_2=df[1][1], dropout=df[1][2], l2=df[1][3], optimizer=optimizer, activation_layer=df[1][6])

history_full = model.fit(train_ds,
              validation_data=val_ds,
              epochs=1000,
              batch_size=200)

# See dataframe of history
hist_full = pd.DataFrame(history_full.history)
hist_full['epoch'] = history_full.epoch
hist_full.tail()

# plot metrics saving
def plot_history(history,early_stop=False):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    #Loss (Mean Squared Logarithmic Error)
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Logarithmic Error')
    plt.plot(hist['epoch'], hist['loss'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_loss'],
            label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()
    plt.show()
    if early_stop==True:
        plt.savefig("plots/loss_mean_squared_logarithmic_error_early.png")
    else:
        plt.savefig("plots/loss_mean_squared_logarithmic_error_full.png")
    plt.close()
    #Root mean squared error
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Root Mean Squared Error')
    plt.plot(hist['epoch'], hist['root_mean_squared_error'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_root_mean_squared_error'],
            label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()
    plt.show()
    if early_stop==True:
        plt.savefig("plots/RootMeanSquaredError_early.png")
    else:
        plt.savefig("plots/RootMeanSquaredError_full.png")
    plt.close()
    #Mean Square Error [$MPG^2$]
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
            label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()
    if early_stop==True:
        plt.savefig("plots/mse_early.png")
    else:
        plt.savefig("plots/mse_full.png")
    plt.close()
    #Mean Abs Error [MPG]
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
            label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()
    plt.show()
    if early_stop==True:
        plt.savefig("plots/mae_early.png")
    else:
        plt.savefig("plots/mae_full.png")
    plt.close()
    #Mean Abs Percentage Error [mape]
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Percentage Error [mape]')
    plt.plot(hist['epoch'], hist['mape'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_mape'],
            label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()
    plt.show()
    if early_stop==True:
        plt.savefig("plots/mape_full.png")
    else:
        plt.savefig("plots/mape_early.png")
    plt.close()
    #Cosine proximity
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Cosine proximity')
    plt.plot(hist['epoch'], hist['cosine_proximity'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_cosine_proximity'],
            label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()
    plt.show()
    if early_stop==True:
        plt.savefig("plots/cosine_early.png")
    else:
        plt.savefig("plots/cosine_full.png")
    plt.close()

# Plotting history of full epochs (1000)
plot_history(history_full)


# IF YOU WANT TO STOP EARLIER THE EPOCHS, WHEN X TIMES THE IMPROVEMENT IS REPEATED IN VALIDATION

# The patience parameter is the amount of epochs to check for improvement
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

history_early_stop = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=1000,
                    batch_size=200, 
                    callbacks=[early_stop])

plot_history(history_early_stop,early_stop=True)


# EVALUATE MODEL WITH TEST
model.evaluate(test_ds)
loss, RootMeanSquaredError, mse, mae, mape, cosine_proximity = model.evaluate(test_ds)

print("Testing set Loss: {:5.2f} Mean Squared Logarithmic Error".format(loss))
print("Testing set Root Mean Squared Error: {:5.2f}".format(RootMeanSquaredError))
print("Testing set Mean Square Error {:5.2f} [$MPG^2$]".format(mse))
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
print("Testing set Mean Abs Percentage Error: {:5.2f} mape".format(mape))
print("Testing set Cosine proximity: {:5.2f}".format(cosine_proximity))


# Example of plotting how good it is with some of the values, in this case MPG
test_predictions = model.predict(test_ds).flatten()

# Scatter plot error
plt.figure()
plt.scatter(test.output, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()
plt.savefig("plots/MPG_predictions_vs_true_values_scatter.png")
plt.close()

# Histogram error
plt.figure()
error = test_predictions - test.output
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()
plt.savefig("plots/MPG_predictions_vs_true_values_hist.png")
plt.close()
