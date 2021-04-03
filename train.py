import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve, auc, fbeta_score
from sklearn.model_selection import train_test_split
from opt import RAdam
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,4,5,6'

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import efficientnet.keras as efn
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import BatchNormalization, Input, Dense, MaxPooling2D, Conv2D, Flatten, Concatenate, Dropout
from keras.layers.core import Activation, Layer
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator

import keras.backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))



print("Importing Dataset")

data_dir = '/home/dados4t/DataChallenge2/'

images_hjy = np.load(os.path.join(data_dir,'images_hjy_normalized.npy'))
images_vis = np.load(os.path.join(data_dir,'images_efn_vis.npy'))
pad = np.zeros((images_hjy.shape[0],images_hjy.shape[1],images_hjy.shape[2],1), dtype="float32")
is_lens = np.load(os.path.join(data_dir,'Y.npy'))

X_train_hjy, X_test_hjy, Y_train, Y_test = train_test_split(np.concatenate([images_hjy[:,:,:,2:],pad,pad], axis=-1), is_lens, test_size = 0.10, random_state = 7)
X_train_vis, X_test_vis, Y_train, Y_test = train_test_split(images_vis, is_lens, test_size = 0.10, random_state = 7)
del images_vis
del images_hjy
print(X_train_hjy.shape)
print(X_train_vis.shape)


print("Building Model")

inp_hjy = Input((66,66,3))
efn_arc_hjy = efn.EfficientNetB2(input_tensor = inp_hjy, weights='imagenet')

for layer in efn_arc_hjy.layers:
    efn_arc_hjy.get_layer(layer.name).name = layer.name + "_y"


inp_vis = Input((200,200,3))
efn_arc_vis = efn.EfficientNetB2(input_tensor = inp_vis, weights='imagenet')

for layer in efn_arc_vis.layers:
    efn_arc_vis.get_layer(layer.name).name = layer.name + "_vis"


concat = Concatenate()([efn_arc_vis.layers[-2].output, efn_arc_hjy.layers[-2].output])

y_hat = Dense(2,activation="softmax")(concat)


model = Model([efn_arc_vis.input, efn_arc_hjy.input], y_hat)

#multigpu
model = multi_gpu_model(model, gpus=5)

model.compile(loss = 'categorical_crossentropy', optimizer=RAdam(),metrics = ['accuracy'])



print("Training Model")

model_name = "efn02_vis_y.hdf5"
batch_size = 35 * 5
check = ModelCheckpoint("final_model/" + model_name, monitor="val_loss", verbose=1, save_best_only=True)

gen = ImageDataGenerator(
		rotation_range=180,
		zoom_range=0.20,
		vertical_flip = True,
    horizontal_flip=True,
		fill_mode="nearest")

def gen_flow_for_two_inputs(X1, X2, y):
    genX1 = gen.flow(X1,y,  batch_size=batch_size,seed=1)
    genX2 = gen.flow(X2, batch_size=batch_size,seed=1)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            #Assert arrays are equal - this was for peace of mind, but slows down training
            #np.testing.assert_array_equal(X1i[0],X2i[0])
            yield [X1i[0], X2i], X1i[1]

gen_flow = gen_flow_for_two_inputs(X_train_vis, X_train_hjy, Y_train)

history = model.fit_generator( gen_flow, epochs = 500,  
            verbose = 1, validation_data= ([X_test_vis, X_test_hjy], Y_test), callbacks=[check], 
            steps_per_epoch = X_train_hjy.shape[0] // batch_size)


print("Getting Statistics")

print("Training Statistics")
pred = model.predict([X_train_vis, X_train_hjy])
    
fig = plt.figure(figsize = (40,20))
fig.suptitle(model_name[:-5])

beta = np.sqrt(0.001)
    
plt.subplot(2,3,1)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], thresh = roc_curve(Y_train[:, i], pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
optimal = np.argmin(np.sqrt(fpr[1]**2 + (1-tpr[1])**2))
optimal_thresh = thresh[optimal]

FB = fbeta_score(Y_train[:,1], 1.*(pred[:,1] > optimal_thresh), average=None, beta=beta)
FB = FB.max()

lw = 2
colors = ['darkblue','darkorange']
classes = ['~lens','lens']
for i in range(2):
    plt.plot(fpr[i], tpr[i], color=colors[i],
             lw=lw, label=f'{classes[i]} (area = %0.2f)' % roc_auc[i])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
fpr_tresh = fpr[1]
tpr_tresh = tpr[1]
plt.plot(fpr_tresh[optimal], tpr_tresh[optimal], '*', label = 'opt -{}'.format(optimal_thresh))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Training ROC - FBeta {FB}')
plt.legend(loc="lower right")


print("Test Statistics")

pred = model.predict([X_test_vis, X_test_hjy])
    
plt.subplot(2,3,2)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
optimal = np.argmin(np.sqrt(fpr[1]**2 + (1-tpr[1])**2))
optimal_thresh = thresh[optimal]

FB = fbeta_score(Y_test[:,1], 1.*(pred[:,1] > optimal_thresh), average=None, beta=beta)
FB = FB.max() 

lw = 2
colors = ['darkblue','darkorange']
classes = ['~lens','lens']
for i in range(2):
    plt.plot(fpr[i], tpr[i], color=colors[i],
             lw=lw, label=f'{classes[i]} (area = %0.2f)' % roc_auc[i])
fpr_tresh = fpr[1]
tpr_tresh = tpr[1]
plt.plot(fpr_tresh[optimal], tpr_tresh[optimal], '*', label = 'opt - {}'.format(optimal_thresh))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Test ROC - FBeta {FB}')
plt.legend(loc="lower right")


print("Best Model Statistics")

model.load_weights(model_name)
pred = model.predict([X_test_vis, X_test_hjy])
    
plt.subplot(2,3,3)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

optimal = np.argmin(np.sqrt(fpr[1]**2 + (1-tpr[1])**2))
optimal_thresh = thresh[optimal]

FB = fbeta_score(Y_test[:,1], 1.*(pred[:,1] > optimal_thresh), average=None, beta=beta)
FB = FB.max()
    
lw = 2
colors = ['darkblue','darkorange']
classes = ['~lens','lens']
for i in range(2):
    plt.plot(fpr[i], tpr[i], color=colors[i],
             lw=lw, label=f'{classes[i]} (area = %0.2f)' % roc_auc[i])
fpr_tresh = fpr[1]
tpr_tresh = tpr[1]
plt.plot(fpr_tresh[optimal], tpr_tresh[optimal], '*', label = 'opt - {}'.format(optimal_thresh))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Test ROC (Loading Best Model) - FBeta {FB}')
plt.legend(loc="lower right")
    

print("Else Statistics")

plt.subplot(2,3,4)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.subplot(2,3,5)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')



plt.savefig(f"final_model/{model_name[:-5]}.png")