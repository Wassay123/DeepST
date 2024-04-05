import os
import math
import time
import numpy as np
import pickle

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from deepst.models.STResNet import stresnet
from deepst.datasets import BikeNYC
from deepst.config import Config
from deepst import metrics

# Setting global parameters
DATAPATH = Config().DATAPATH  # Set your own data path with the global environmental variable DATAPATH
nb_epoch = 50   # Number of epochs at the training stage
nb_epoch_cont = 100  # Number of epochs at the training (cont) stage
batch_size = 64  # Batch size
T = 24  # Number of time intervals in one day
lr = 0.0002  # Learning rate
len_closeness = 3  # Length of closeness-dependent sequence
len_period = 4  # Length of period-dependent sequence
len_trend = 4  # Length of trend-dependent sequence
nb_residual_unit = 4   # Number of residual units
nb_flow = 2  # Number of flow types: new-flow and end-flow
days_test = 10  # Number of days in the test set
len_test = T * days_test  # Length of the test set
map_height, map_width = 16, 8  # Grid size
nb_area = 81  # Number of grid-based areas for NYC Bike data
m_factor = math.sqrt(1. * map_height * map_width / nb_area)  # Factor for modifying the final RMSE
print('factor: ', m_factor)
path_result = 'RET'  # Directory for result files
path_model = 'MODEL'  # Directory for model checkpoints
np.random.seed(1337)  # for reproducibility

# Creating result and model directories if they don't exist
if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)

# function to build the STResNet Model
def build_model(external_dim):
    c_conf = (len_closeness, nb_flow, map_width, map_height) if len_closeness > 0 else None
    p_conf = (len_period, nb_flow, map_width, map_height) if len_period > 0 else None
    t_conf = (len_trend, nb_flow, map_width, map_height) if len_trend > 0 else None

    model = stresnet(c_conf=c_conf,
                     p_conf=p_conf, 
                     t_conf=t_conf,
                     external_dim=external_dim, 
                     nb_residual_unit=nb_residual_unit)
    
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    model.summary()

    return model

def main():

    print("loading data...")
    X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = BikeNYC.load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
        preprocess_name='preprocessing.pkl', meta_data=True)

    print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])

    print('=' * 10)
    print("compiling model...")
    print("**at the first time, it takes a few minutes to compile if you use [Theano] as the backend**")

    # Build the STResNet Model
    model = build_model(external_dim)
    hyperparams_name = 'c{}.p{}.t{}.resunit{}.lr{}'.format(len_closeness, len_period, len_trend, nb_residual_unit, lr)
    fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))

    # Setting up callbacks for early stopping and model checkpoint
    early_stopping = EarlyStopping(monitor='val_rmse', patience=5, mode='min')
    model_checkpoint = ModelCheckpoint(fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

    print('=' * 10)
    print("training model...")

    # Train the model
    history = model.fit(X_train, Y_train,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        validation_split=0.1,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)
    
    # Saving model weights and training history
    model.save_weights(os.path.join('MODEL', '{}.h5'.format(hyperparams_name)), overwrite=True)
    pickle.dump((history.history), open(os.path.join(path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))

    print('=' * 10)
    print('evaluating using the model that has the best loss on the valid set')

    # Loading the best model and evaluating on the training set
    model.load_weights(fname_param)
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[0] // 48, verbose=0)
    print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' % (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2. * m_factor))

    # Evaluating the model on the test set
    score = model.evaluate(X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' % (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2. * m_factor))

    print('=' * 10)
    print("training model (cont)...")\
    
    fname_param = os.path.join('MODEL', '{}.cont.best.h5'.format(hyperparams_name))
    model_checkpoint = ModelCheckpoint(fname_param, monitor='rmse', verbose=0, save_best_only=True, mode='min')
    
    # Training the model on additional epochs (continuation)
    history = model.fit(X_train, Y_train, 
                        epochs=nb_epoch_cont, 
                        verbose=1, 
                        batch_size=batch_size, 
                        callbacks=[model_checkpoint], 
                        validation_data=(X_test, Y_test))
    
    pickle.dump((history.history), open(os.path.join(path_result, '{}.cont.history.pkl'.format(hyperparams_name)), 'wb'))

if __name__ == "__main__":
    main()