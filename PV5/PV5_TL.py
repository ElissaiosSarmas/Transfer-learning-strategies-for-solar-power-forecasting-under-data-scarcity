# Read CSV
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# Create dataframe
merged = pd.read_csv(r'PV5.csv')
merged.drop('Unnamed: 0',axis=1,inplace=True)
merged['datetime'] = pd.to_datetime(merged['datetime'])
merged['timestamp'] = pd.to_datetime(merged['timestamp']).dt.hour

# Select non-correlated variables and perform One-Hot-Encoding to Months and Sin-Cos Similarities to Hours
data = merged[['Produzida','year','month','day','timestamp','Temperature','Humidity','Solar w/m2']]
data = pd.get_dummies(data, columns=['month'])
data.loc[:,'sin_hour'] = np.sin(2*np.pi*data['timestamp']/24)
data.loc[:,'cos_hour'] = np.cos(2*np.pi*data['timestamp']/24)

print("Total Dataset =", len(data))



# Shaping data for LSTM input
def split_sequences(sequences, n_steps, n_outputs, only_production):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern 
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        if only_production==True:
            seq_x, seq_y = sequences[i:end_ix, -1], sequences[end_ix:(end_ix+n_outputs), -1]
        else:
            seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:(end_ix+n_outputs), -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def unique_shapes(x, y, lag_, n_features_, num_of_outputs_, only_production):
    uniuqe_shapes = []
    for k in range(len(x)):
        if only_production==True:
            if (x[k].shape == (lag,)) & (y[k].shape == (num_of_outputs_,)):
                uniuqe_shapes.append(k)
        else:
            if (x[k].shape == (lag_, n_features_)) & (y[k].shape == (num_of_outputs_,)):
                uniuqe_shapes.append(k)       
    x = x[uniuqe_shapes]
    y = y[uniuqe_shapes]
    x = np.stack(x)
    y = np.stack(y)
    return x, y



# Select the columns that you want to use as features
cols = ['Temperature',
        'Humidity',
        'Solar w/m2', 
        'month_1',
        'month_2',       
        'month_3',
        'month_4',        
        'month_5',
        'month_6',        
        'month_7',
        'month_8',        
        'month_9',
        'month_10',        
        'month_11',
        'month_12',
        'sin_hour',
        'cos_hour',
        'timestamp', 
        'Produzida']
# Set to True if using only the production, else to False
only_production = False
# Splitting factor for training set and test set
split = 0.582

# Select the lag variable, the number of features (must be same with cols selected) and the horizon
lag = 5
n_features = len(cols)
num_of_outputs = 1



# Scale data seperately for preventing data leakage
from sklearn.preprocessing import MinMaxScaler,StandardScaler

if only_production == True:
    data_ = data_['Produzida']
    train = data_.iloc[:int(len(data_)*split_),]
    test = data_.iloc[int(len(data_)*split_):,]
    scaler = MinMaxScaler(feature_range=(0, 1))
    train = scaler.fit_transform(train.values.reshape(-1, 1))
    test = scaler.fit_transform(test.values.reshape(-1, 1))
else:
    data = data[cols]
    train = data.iloc[:int(len(data)*split),:]
    test = data.iloc[int(len(data)*split):,]
    scaler = MinMaxScaler(feature_range=(0,1))
    train = scaler.fit_transform(train)
    test = scaler.fit_transform(test)




# Create the input for LSTM: x(batch_size, lag, features), y(batch_size,)
x_train, y_train = split_sequences(train, n_steps=lag, n_outputs=num_of_outputs, only_production=only_production)
x_test, y_test = split_sequences(test, n_steps=lag, n_outputs=num_of_outputs, only_production=only_production)

x_train, y_train = unique_shapes(x_train, y_train, lag, n_features, num_of_outputs, only_production=only_production)
x_test, y_test = unique_shapes(x_test, y_test, lag, n_features, num_of_outputs, only_production=only_production)

# Reshape for only_production case
if only_production==True:
    x_train = x_train.reshape((x_train.shape[0], lag, 1))
    x_test = x_test.reshape((x_test.shape[0], lag, 1))

# Print the shapes
print("Size of Train / Test =", x_train.shape, y_train.shape, x_test.shape, y_test.shape)


def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred))))

def createModel():

    # LSTM Model Architecture
    from tensorflow.keras.models import load_model
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam

    # Load Base model
    model = load_model(r'../PV1_base/PV1_base.h5')

    # Compile model
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
    model.summary()

    return model


def fit_predict_stats(model):

    from tensorflow.keras.models import load_model
    from tensorflow.keras.callbacks import EarlyStopping

    # Early stopping property
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

    # Fit the model
    history = model.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=128, verbose=1, shuffle=True, callbacks=[es]).history

    # Save the model
    model.save('PV5_TL.h5')

    #Load the model
    model = load_model(r'PV5_TL.h5')

    # summarize history for MAE and MSE
    # plt.plot(history['loss'])
    # plt.plot(history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('Model MSE')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')

    # plt.figure()
    # plt.plot(history['mae'])
    # plt.plot(history['val_mae'])
    # plt.title('Model MAE')
    # plt.ylabel('MAE')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')

    # Metrics on scaled data
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    y_pred = model.predict(x_train)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    mae = mean_absolute_error(y_train, y_pred)
    print('Train Scaled RMSE: {}'.format(rmse))
    print('Train Scaled MAE: {}'.format(mae))
    print('Train Scaled R2 Score: ', r2_score(y_train, y_pred)*100)

    y_pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print('Test Scaled RMSE: {}'.format(rmse))
    print('Test Scaled MAE: {}'.format(mae))
    print('Test Scaled R2 Score: ',r2_score(y_test, y_pred)*100)

    # Metrics on original data
    true = []
    hat = []
    range_ = [0]
    # range_ = list(range(6))

    for i,j in zip([[x_train,y_train],[x_test,y_test]],['Train','Test']):
        # make a prediction
        yhat = model.predict(i[0])
        if yhat.shape == (yhat.shape[0],):
            yhat = yhat.reshape((yhat.shape[0],1))
        y_hat = []
        for k in range(len(yhat)):
            if k == 0:
                for l in range_:
                    y_hat.append(yhat[k,l])
            else:
                y_hat.append(yhat[k,-1])
        
        y_hat = np.stack(y_hat)
        y_hat = y_hat.reshape((y_hat.shape[0],1))
        
        i[0] = i[0].reshape((i[0].shape[0],lag,n_features))
        
        x_hat = []
        for k in range(len(i[0])):
            if k == 0:
                x_hat.append(i[0][k])
            elif k!= 0:
                x_hat.append(i[0][k][-1,:])
        
        x_hat = np.vstack(x_hat)
        
        initial_x_hat_shape = x_hat.shape[0]
        initial_y_hat_shape = y_hat.shape[0]
        
        print(x_hat.shape)
        print(y_hat.shape)
        
        if x_hat.shape[0]-y_hat.shape[0] != 0.0:
            if x_hat.shape[0] > y_hat.shape[0]:
                for k in range(x_hat.shape[0]-y_hat.shape[0]):
                    y_hat = np.insert(y_hat, 0, y_hat[0,0], axis=0)
                    added_values = True
            elif x_hat.shape[0] < y_hat.shape[0]:
                y_hat = y_hat[-int(x_hat.shape[0]-y_hat.shape[0]):,:]
                added_values = False
        
        print(x_hat.shape)
        print(y_hat.shape)
        
        # invert scaling for forecast
        if only_production==True:
            inv_yhat = np.concatenate((x_hat[:,:-1],y_hat), axis=1)
            inv_yhat = scaler.inverse_transform(inv_yhat)
        else:    
            inv_yhat = np.concatenate((x_hat[:,:-1],y_hat), axis=1)
            inv_yhat = scaler.inverse_transform(inv_yhat)
            inv_yhat = inv_yhat[:,-1]
            
        # invert scaling for actual
        y_true = []
        for k in range(len(i[1])):
            if k ==0:
                for l in range_:
                    y_true.append(i[1][k,l])
            else:
                y_true.append(i[1][k,-1])

        y_true = np.stack(y_true)
        y_true = y_true.reshape((y_true.shape[0],1))
        print(y_true.shape)

        initial_y_true_shape = y_true.shape[0]
        
        if x_hat.shape[0]-y_true.shape[0] != 0.0:
            if x_hat.shape[0] > y_true.shape[0]:
                for k in range(x_hat.shape[0]-y_true.shape[0]):
                    y_true = np.insert(y_true, 0, y_true[0,0], axis=0)
                    added_values = True
            elif x_hat.shape[0] < y_true.shape[0]:
                y_true = y_true[-int(x_hat.shape[0]-y_true.shape[0]):,:]
                added_values = False
                
        if only_production==True:
            inv_y = np.concatenate((x_hat[:,:-1],y_true), axis=1)
            inv_y = scaler.inverse_transform(inv_y)
        else:
            inv_y = np.concatenate((x_hat[:,:-1],y_true), axis=1)
            inv_y = scaler.inverse_transform(inv_y)
            inv_y = inv_y[:,-1]
        
        true.append(inv_y)
        hat.append(inv_yhat)
        
        # calculate RMSE
        rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
        print('Test RMSE: %.3f' % rmse)
        # calculate MAE
        mae = mean_absolute_error(inv_y, inv_yhat)
        print('Test MAE: %.3f' % mae)
        # calculate R2
        r2 = r2_score(inv_y, inv_yhat)*100
        print('Test R2 Score: ',r2)
        # Calculate MAPE
        mape = mean_absolute_percentage_error(inv_y, inv_yhat)
        print('MAPE', mape)
        # Calculate MBE
        mbe = np.mean(inv_yhat - inv_y)
        print('Test MBE', mbe)
        # Calculate nRMSE
        nRMSE = rmse / np.mean(inv_y)
        print('nRMSE', nRMSE)


    mae_list.append(mae)
    rmse_list.append(rmse)
    r_square_list.append(r2)
    mape_list.append(mape)
    mbe_list.append(mbe)
    nRMSE_list.append(nRMSE)

mae_list = []
rmse_list = []
r_square_list = []
mape_list = []
mbe_list = []
nRMSE_list = []
for i in range(20):
    model = createModel()
    fit_predict_stats(model)


# print(mae_list)
print(rmse_list)
# print(r_square_list)
# print(mape_list)
# print(mbe_list)
# print(nRMSE_list)

print(np.mean(mae_list))
print(np.mean(rmse_list))
print(np.mean(r_square_list))
print(np.mean(mape_list))
print(np.mean(mbe_list))
print(np.mean(nRMSE_list))