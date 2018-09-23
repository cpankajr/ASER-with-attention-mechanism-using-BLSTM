
from keras import Sequential
from keras.layers import (Activation, Convolution2D, Dense, Dropout, Flatten,MaxPooling2D, BatchNormalization, LSTM ,Bidirectional,Reshape,Input,
	RepeatVector)
import _pickle as cPickle 
from sklearn.cross_validation import train_test_split
from keras.layers.wrappers import TimeDistributed
import tensorflow as tf
from time import time
from keras.optimizers import SGD, Adadelta, Nadam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from atten_layer import AttentionLayer
from keras.models import Model
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
def get_model(summary=False):
    """ Return the Keras model of the network
    """
    model = Sequential()
    
    inputs = Input(shape=(300, 40,3))
    # 1st layer group
    CNN1=Convolution2D(128, 5, 3, activation='relu', border_mode='same', name='conv1', subsample=(1, 1))(inputs)
    MAX_POOL=MaxPooling2D(pool_size=(1, 4),border_mode='valid', name='pool1')(CNN1)
    BN1=BatchNormalization()(MAX_POOL)
    CNN2=Convolution2D(256, 5,3, activation='relu', 
    	border_mode='same', name='conv3a',
    	subsample=(1, 1))(BN1)
    MAX_POOL2=MaxPooling2D(pool_size=(1, 2),border_mode='valid', name='pool2')(CNN2)
    BN2=BatchNormalization()(MAX_POOL2)
    CNN3=Convolution2D(256, 5,3, activation='relu', 
    	border_mode='same', name='conv3b',
    	subsample=(1, 1))(BN2)
    DROP1=Dropout(.5)(CNN3)
    BN3=BatchNormalization()(DROP1)

    CNN4=Convolution2D(256, 5,3, activation='relu', 
    	border_mode='same', name='conv3c',
    	subsample=(1, 1))(BN3)
    BN4=BatchNormalization()(CNN4)
    CNN5=Convolution2D(256, 5,3, activation='relu', 
    	border_mode='same', name='conv3d',
    	subsample=(1,1))(BN4)
    BN5=BatchNormalization()(CNN5)
    DROP2=Dropout(.5)(BN5) 
    # FLAT1=Reshape((300, 5*512))(DROP2)
    # FLAT1=Flatten()(DROP2)
    # RESHAPE=Reshape((200,3,256*5))(DROP2)
    # FC layers group
    TD=TimeDistributed(Flatten(), name="Flatten")(DROP2)

    DENSE1=Dense(768, activation='linear', name='fc6')(TD)
    # print(FLAT1.get_shape())
    #RESHAPE2=Reshape((200, 768))(FLAT1)
    # RV=RepeatVector(200)(DENSE1)
    BLSTM=Bidirectional(LSTM(128,return_sequences=True,unit_forget_bias=True))(DENSE1)
    # print(BLSTM.get_shape())
    gru = AttentionLayer(name='attention')(BLSTM)
    # print(gru.get_shape())    
    # print(tf.shape(gru))
    # Bidirectional(LSTM(128)))]
    DROP3=Dropout(0.5)(gru)
    # RESHAPE2=Reshape((None, None))(DROP3)
    DENSE2=Dense(64, activation='relu', name='fc7')(DROP3)
    BN3=BatchNormalization()(DENSE2)
    DROP4=Dropout(0.5)(BN3)
    DENSE3=Dense(7, activation='softmax')(DROP4)
    print(DENSE3.get_shape())
    model = Model(input=inputs, output=DENSE3)
    if summary:
    	print(model.summary())
    return model
if __name__ == '__main__':
	tf.device('/cpu:0')
	model=get_model(summary=True)
	f = open('C:/Users/cpank/Desktop/emodb.pkl', 'rb')
	train_data,train_label,em_label = cPickle.load(f)
	# train_data = train_data.reshape(339,300, 40,3)
	y_label = np_utils.to_categorical(train_label, 7)
	X_train, X_val, y_train, y_val = train_test_split(train_data, y_label, test_size=0.2, random_state=4)
	print(X_val.shape)
	nadam = Nadam(lr=1e-04, beta_1=0.9, beta_2=0.999, epsilon=1e-08,schedule_decay=0.004)
	sgd = SGD(lr=0.0001, decay=1e-5, momentum=0.9, nesterov=True)
	model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])
	tensorboard = TensorBoard(log_dir="./logs/{}".format(time()), batch_size=50,
                          write_graph=True, write_grads=True, write_images=True)
	t0 = time()
	model.fit(X_train,y_train, callbacks=[tensorboard], batch_size=50,nb_epoch = 50,shuffle=True,
		validation_data=(X_val,y_val))
	t1 = time()
	print("Training completed in " + str((t1 - t0) / 3600) + " hours")
	model.save_weights('cnn_lstm_weights.h5')
	print("Weights Saved Successfully...")
	model.save('cnn_lstm_model.h5')
	print("Model Saved Successfully...")
	# f = open('C:/Users/cpank/Desktop/emodb_test.pkl', 'rb')
 #    test_data,test_label,em_test_label = cPickle.load(f)
	# # y_test_label = np_utils.to_categorical(test_label, 7)
 # #    score = model.evaluate(test_data, y_test_label, verbose=0)
 # #    print('Test loss:', score[0])
	# # print('Test accuracy:', score[1])