
# coding: utf-8

# In[36]:


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


# In[46]:


#! head -n 10000000 train.csv > traintrim.csv
#! head -n 10000000 train.txt > traintrim.txt


# In[47]:


#! head -n 10000000 test.csv > testtrim.csv
#! head -n 10000000 test.txt > testtrim.txt


# In[39]:


preprocess = False


# In[40]:


def encode_column(col):
    encoder = preprocessing.LabelEncoder()
    #small_vals = train.groupby(col).count()[0].where(lambda x: x <= 1).dropna().apply(lambda x: '1').to_dict()
    #train.iloc[:,col] = train.iloc[:,col].apply(lambda x : small_vals.get(x,x))
    set_ = train.loc[:, col].values
    c = Counter(set_)
    small_vals = dict(zip(list(dict(filter(lambda x: x[1] <= 3, c.most_common())).keys()), itertools.repeat('1') ))
    train.iloc[:,col] = train.iloc[:,col].apply(lambda x : small_vals.get(x,x))
    encoder.fit(train.loc[:, col].fillna('nan').values)
    return encoder
def transform_column(col):
    encoder = encoders[col-14]
    return encoder.transform(train.loc[:, col].fillna('nan').values)
def encode_test_column(col): 
    encoder = encoders[col-13]
    dic = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    return test.loc[:, col].fillna('nan').map(dic).fillna(dic.get('nan',0)).values


# In[41]:


if preprocess: 
    print('Import')
    train_gen = pd.read_csv("train.txt", sep='\t', lineterminator='\n', header=None, engine='c', chunksize = 100000)
    train = pd.concat([chunk for chunk in tqdm(train_gen)])
    test_gen = pd.read_csv("test.txt", sep='\t', lineterminator='\n', header=None, engine='c', chunksize = 100000)
    test = pd.concat([chunk for chunk in tqdm(test_gen)])
    print('Transform')
    from sklearn import preprocessing
    pool = Pool()
    #encoders = pool.map(encode_column, tqdm(range(14,40)))
    encoders = list(map(encode_column, tqdm(range(14,40))))
    #transformed_cols = pool.map(transform_column, tqdm(range(14,40)))
    transformed_cols = list(map(transform_column, tqdm(range(14,40))))
    for col in tqdm(range(14,40)):
        train.loc[:, col] = transformed_cols[col-14]
    #transformed_test_cols = pool.map(encode_test_column, tqdm(range(13,39)))
    transformed_test_cols = list(map(encode_test_column, tqdm(range(13,39))))
    for col in tqdm(range(13,39)):
        test.loc[:, col] = transformed_test_cols[col-13]
    print('filna')
    train = train.fillna(0)#.loc[:,list(range(14))]
    test = test.fillna(0)#.loc[:,list(range(13))].values
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
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

