
# coding: utf-8

# In[ ]:


from __future__ import absolute_import, division, print_function
import pandas as pd
from tqdm import tqdm
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

from tensorflow.python.client import device_lib
device_lib.list_local_devices()
import pickle 
from multiprocessing import Pool
import os
import time
import datetime
import matplotlib as plt
import seaborn as sns
from collections import Counter
import itertools
from sklearn.utils import class_weight
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing


# In[ ]:


#! head -n 10000000 train.csv > traintrim.csv
#! head -n 10000 train.txt > traintrim.txt
#! head -n 10000000 test.csv > testtrim.csv
#! head -n 10000 test.txt > testtrim.txt


# In[ ]:


def encode_column(col, df):
    encoder = preprocessing.LabelEncoder()
    #small_vals = train.groupby(col).count()[0].where(lambda x: x <= 1).dropna().apply(lambda x: '1').to_dict()
    #train.iloc[:,col] = train.iloc[:,col].apply(lambda x : small_vals.get(x,x))
    set_ = df.loc[:, col].values
    c = Counter(set_)
    small_vals = dict(zip(list(dict(filter(lambda x: x[1] <= 5, c.most_common())).keys()), itertools.repeat('1') ))
    df.loc[:,col] = df.loc[:,col].apply(lambda x : small_vals.get(x,x))
    encoder.fit(df.loc[:, col].dropna().values)
    return encoder
def transform_column(col, df):
    encoder = encoders[col]
    return encoder.transform(df.loc[:, col].dropna().values)
def encode_test_column(col, df): 
    encoder = encoders[col+1]
    dic = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    return df.loc[df.loc[:, col].dropna().index, col].map(dic).values


# In[ ]:


preprocess = True


# In[ ]:


def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]


# In[ ]:


if preprocess: 
    print('Import')
    train_gen = pd.read_csv("traintrim.txt", sep='\t', lineterminator='\n', header=None, engine='c', chunksize = 100000)
    train = pd.concat([chunk for chunk in tqdm(train_gen)])
    test_gen = pd.read_csv("testtrim.txt", sep='\t', lineterminator='\n', header=None, engine='c', chunksize = 100000)
    test = pd.concat([chunk for chunk in tqdm(test_gen)])
    print(np.mean((train.count()/len(train)).values), np.mean((test.count()/len(test)).values))
    print('Transform')
    encoders = {x: encode_column(x, train) for x in tqdm(train.loc[:, train.columns > 13].columns)}
    transformed_cols = {x: transform_column(x, train) for x in  tqdm(train.loc[:, train.columns > 13].columns)}
    for col in tqdm(train.loc[:, train.columns > 13].columns):
        train.loc[train.loc[:, col].dropna().index, col] = transformed_cols[col]
    print('Learn')
    predictors = dict()
    not_nan_cols_dict = dict()
    for col in tqdm(train.drop(0,axis=1).columns):
        not_nan_cols = train.drop(0,axis=1).loc[train.loc[:, col].isna()].count()/len(train.loc[train.loc[:, col].isna()]) > .80
        not_nan_cols_dict[col] = list(train.drop(0,axis=1).loc[:,not_nan_cols].columns)
        train_nonan = train.loc[:, np.append(np.array(not_nan_cols_dict[col]), col)].dropna()#.drop(0,axis=1)
        if len(train_nonan.drop(col, axis = 1).values[0]) > 0:
            x_nonan = train_nonan.drop(col, axis = 1).values
            y_nonan = train.loc[train_nonan.index, col].values
            #splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
            #for train_index, test_index in splitter.split(x_nonan, y_nonan):
            #    x_train_nonan, x_test_nonan = x_nonan.loc[train_index].values, x_nonan.loc[test_index].values
            #    y_train_nonan, y_test_nonan = y_nonan.loc[train_index].values, y_nonan.loc[test_index].values
            if col in list(range(14)):
                predictors[col] = RandomForestRegressor(max_depth=3, random_state=0, n_estimators=50, n_jobs=10).fit(x_nonan, y_nonan)
            elif col in list(range(14,40)):
                y_nonan=y_nonan.astype('int')
                predictors[col] = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=0, n_jobs=10).fit(x_nonan, y_nonan)
    with open("predictorstrain.p","wb") as filehandler:
        pickle.dump(predictors, filehandler, protocol=4)
    print('Predict')
    with open("predictorstrain.p","rb") as filehandler:
        predictors = pickle.load(filehandler)
    for col in tqdm(predictors.keys()):
        not_nan_col_lines = train.loc[train.loc[:,col].isna(), not_nan_cols_dict[col]].dropna()
        for index in tqdm(chunks(not_nan_col_lines.index, 1000000)):
            train.loc[index, col] = predictors[col].predict(not_nan_col_lines.loc[index,:].values)
    print(np.mean((train.count()/len(train)).values), np.mean((test.count()/len(test)).values))
    print('Transform')
    transformed_test_cols = {x: encode_test_column(x, test) for x in tqdm(test.loc[:, test.columns > 12].columns)}
    for col in tqdm(range(13,39)):
        test.loc[test.loc[:, col].dropna().index, col] = transformed_test_cols[col]
    print('Learn')
    predictors = dict()
    not_nan_cols_dict = dict()
    for col in tqdm(test.columns):
        not_nan_cols = test.loc[test.loc[:, col].isna()].count()/len(test.loc[test.loc[:, col].isna()]) > .80
        not_nan_cols_dict[col] = list(test.loc[:,not_nan_cols].columns)
        test_nonan = test.loc[:, np.append(np.array(not_nan_cols_dict[col]), col)].dropna()#.drop(0,axis=1)
        if len(test_nonan.drop(col, axis = 1).values[0]) > 0:
            x_nonan = test_nonan.drop(col, axis = 1).values
            y_nonan = test.loc[test_nonan.index, col].values
            #splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
            #for train_index, test_index in splitter.split(x_nonan, y_nonan):
            #    x_train_nonan, x_test_nonan = x_nonan.loc[train_index].values, x_nonan.loc[test_index].values
            #    y_train_nonan, y_test_nonan = y_nonan.loc[train_index].values, y_nonan.loc[test_index].values
            if col in list(range(13)):
                predictors[col] = RandomForestRegressor(max_depth=3, random_state=0, n_estimators=50, n_jobs=10).fit(x_nonan, y_nonan)
            elif col in list(range(13,40)):
                y_nonan=y_nonan.astype('int')
                predictors[col] = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=0, n_jobs=10).fit(x_nonan, y_nonan)
    with open("predictorstest.p","wb") as filehandler:
        pickle.dump(predictors, filehandler, protocol=4)
    print('Predict')
    with open("predictorstest.p","rb") as filehandler:
        predictors = pickle.load(filehandler)
    for col in tqdm(predictors.keys()):
        not_nan_col_lines = test.loc[test.loc[:,col].isna(), not_nan_cols_dict[col]].dropna()
        test.loc[not_nan_col_lines.index, col] = predictors[col].predict(not_nan_col_lines)
    print(np.mean((train.count()/len(train)).values), np.mean((test.count()/len(test)).values))
    print('filna')
    train = train.fillna(0)
    test = test.fillna(0)
    print(np.mean((train.count()/len(train)).values), np.mean((test.count()/len(test)).values))

    print('Export')
    
    train.to_csv('train.csv', index=None, header=False)
    test.to_csv('test.csv', index=None, header=False)
    
    #with open("train.p","wb") as filehandler:
    #    pickle.dump(train, filehandler, protocol=4)
    
    #with open("test.p","wb") as filehandler: 
    #    pickle.dump(test, filehandler, protocol=4)


# In[ ]:


print("Import preprocessed CSV")
train_gen = pd.read_csv("train.csv",  header=None, engine='c', chunksize = 100000)
train = pd.concat([chunk for chunk in tqdm(train_gen)])
test_gen = pd.read_csv("test.csv", header=None, engine='c', chunksize = 100000)
test = pd.concat([chunk for chunk in tqdm(test_gen)]).values


# In[ ]:


print("split")
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
X = train.drop(0, axis = 1)
y = train[0]#.values.reshape([-1,1])
#enc = OneHotEncoder(sparse=False)
#y = enc.fit_transform(y)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
for train_index, test_index in splitter.split(X, y):
    X_train, X_test = X.loc[train_index].values, X.loc[test_index].values
    y_train, y_test = y.loc[train_index].values, y.loc[test_index].values
    
del train


# In[ ]:


print(X.shape)
print(y.shape)
print(test.shape)


# In[ ]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(X.shape[1],)),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[ ]:


weights = class_weight.compute_class_weight('balanced',
                                            np.unique(y_train),
                                            y_train)

now = datetime.datetime.now().strftime("%Y%m%d%H%M")


# In[ ]:


checkpoint_path = "training"+now+"/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5)
early_cp = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0.0001,
                              patience=15,
                              verbose=1, mode='auto', restore_best_weights=True)
tboard_cp = keras.callbacks.TensorBoard(log_dir='./Graph/'+now, histogram_freq=0,  
          write_graph=True, write_images=True)
model.save_weights(checkpoint_path.format(epoch=0))


# In[ ]:


history = model.fit(X_train, y_train, 
                    epochs=200, 
                    batch_size=512,  verbose=1,
                    validation_data=(X_test, y_test),
                    callbacks = [cp_callback, early_cp, tboard_cp],
		    class_weight=weights)


# In[ ]:


history_dict = history.history


# In[ ]:


import matplotlib.pyplot as plt

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('val_loss.png', bbox_inches='tight')
# "bo" is for "blue dot"
plt.close()

plt.plot(epochs, acc, 'bo', label='Training acc')
# b is for "solid blue line"
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.legend()


plt.savefig('val_acc.png', bbox_inches='tight')


# In[ ]:


results = model.evaluate(X_test, y_test)


# In[ ]:



#sns.distplot(model.predict(X_test))


# In[ ]:


y_test_pred = pd.DataFrame(model.predict(X_test)).apply(lambda x: int(round(x)), axis = 1).to_frame()
y_test_pred.groupby(0)[0].count()#/y_test_pred.groupby(0)[1].sum()


# In[ ]:


y_pred = model.predict(test)


# In[ ]:


pd.DataFrame(list(zip(list(range(60000000, 60000000+len(y_pred))), y_pred.reshape([1,-1]).tolist()[0])), columns=["Id","Predicted"]).to_csv(str(int(round(results[1]*1000)))+'submission.csv', index=False)

