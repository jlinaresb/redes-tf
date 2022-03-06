# Use scikit-learn to grid search the batch size and epochs
import numpy as np
import pandas as pd


import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt


# load dataset
trainSet = pd.read_csv('audit_data.csv')

# Feature generation: training data MAKE BETTER FEATURE TRANSFORMATIONS
trainSet["output"] = trainSet['Audit_Risk']
data = trainSet.drop(columns=['Audit_Risk', 'Detection_Risk'])

data.drop(data.loc[(data.LOCATION_ID  == 'LOHARU')].index, inplace = True)
data.drop(data.loc[(data.LOCATION_ID  == 'NUH')].index, inplace = True)
data.drop(data.loc[(data.LOCATION_ID  == 'SAFIDON')].index, inplace = True)
data["LOCATION_ID"] = pd.to_numeric(data.LOCATION_ID)

data = data.dropna(axis = 0)

# Train/test split
train, test = train_test_split(data, test_size=0.2)
train, val = train_test_split(train, test_size=0.25)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('output')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

train_ds = df_to_dataset(train)
val_ds = df_to_dataset(val)
test_ds = df_to_dataset(test)

# Scale
def get_scal(feature):
  def minmax(x):
    mini = train[feature].min()
    maxi = train[feature].max()
    return (x - mini)/(maxi-mini)
  return(minmax)

# Scale DEPENDING OF KIND OF FEATURE, for feature coding see: https://medium.com/ml-book/demonstration-of-tensorflow-feature-columns-tf-feature-column-3bfcca4ca5c4

#Numericals
num_c = ['Sector_score', 'PARA_A', 'Score_A', 'Risk_A', 'PARA_B',
       'Score_B', 'Risk_B', 'TOTAL', 'numbers', 'Score_B.1', 'Risk_C',
       'Money_Value', 'Score_MV', 'Risk_D', 'District_Loss', 'PROB', 'RiSk_E',
       'History', 'Prob', 'Risk_F', 'Score', 'Inherent_Risk','CONTROL_RISK']
bucket_c = ['LOCATION_ID']

#Categorical
cat_i_c = ["Risk"]

# Numerical columns
feature_columns = []
for header in num_c:
  scal_input_fn = get_scal(header)
  feature_columns.append(feature_column.numeric_column(header, normalizer_fn=scal_input_fn))

# Bucketized columns
LOCATION_ID = feature_column.numeric_column("LOCATION_ID")
LOCATION_buckets = feature_column.bucketized_column(LOCATION_ID, boundaries=[5, 10, 15, 20, 25, 30, 35, 40, 45])
feature_columns.append(LOCATION_buckets)

# Categorical indicator columns
for feature_name in cat_i_c:
  vocabulary = data[feature_name].unique()
  cat_c = tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)
  one_hot = feature_column.indicator_column(cat_c)
  feature_columns.append(one_hot)

# Categorical embedding columns 
#for feature_name in cat_e_c:
#  vocabulary = data[feature_name].unique()
#  cat_c = tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)
#  embeding = feature_column.embedding_column(cat_c, dimension=50)
#  feature_columns.append(embeding)

# Crossed columns
#vocabulary = data['Sex'].unique()
#Sex = tf.feature_column.categorical_column_with_vocabulary_list('Sex', vocabulary)

#crossed_feature = feature_column.crossed_column([age_buckets, Sex], hash_bucket_size=1000)
#crossed_feature = feature_column.indicator_column(crossed_feature)
#feature_columns.append(crossed_feature)
#len(feature_columns)

# SET UP FOR HYPERPARAMETERS, this code is explained here for regression: https://analyticsindiamag.com/parameter-tuning-tensorboard/
HP_NUM_UNITS1 = hp.HParam('num_units 1', hp.Discrete([4,8,16])) 
HP_NUM_UNITS2 = hp.HParam('num_units 2', hp.Discrete([4,8]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.2, 0.5))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['SGD', 'Adam', 'RMSprop']))
HP_OPTIMIZER_LR = hp.HParam('learning_rate', hp.Discrete([0.0001, 0.001, 0.1]))
HP_L2 = hp.HParam('l2 regularizer', hp.RealInterval(.001,.01))
HP_ACTIVATION_LAYER = hp.HParam('activation_layers', hp.Discrete(['relu', 'sigmoid', 'tanh']))
#Settinf the Metric to RMSE
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

# Create feature layer
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# Define model 
def train_test_model(hparams):
  model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(hparams[HP_NUM_UNITS1], kernel_regularizer=tf.keras.regularizers.l2(hparams[HP_L2]), activation=hparams[HP_ACTIVATION_LAYER]),
    layers.Dropout(hparams[HP_DROPOUT]),
    layers.Dense(hparams[HP_NUM_UNITS2], kernel_regularizer=tf.keras.regularizers.l2(hparams[HP_L2]), activation=hparams[HP_ACTIVATION_LAYER]),
    tf.keras.layers.Dense(1)
  ])
  model.compile(optimizer=optimizer,
                loss='mean_squared_logarithmic_error',
                metrics=['RootMeanSquaredError'])
  model.fit(train_ds,
            validation_data=val_ds,
            epochs=5)
  _, accuracy = model.evaluate(val_ds)
  return accuracy

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
