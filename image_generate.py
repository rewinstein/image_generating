import numpy as np
import cv2
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input, UpSampling2D
from keras import backend as K



batch_size = 1
num_classes = 0

img_w = 100
img_h = 100

img1 = cv2.imread('l1.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('l2.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

temp_arr = []
temp_arr.append(img1)
img1 = np.asarray(temp_arr)
img1 = img1.transpose(2,1,0)


temp_arr = []
temp_arr.append(img2)
img2 = np.asarray(temp_arr)
img2 = img1.transpose(2,1,0)


img1 = np.resize(img1, (img_w, img_h, 1))
img2 = np.resize(img2, (img_w, img_h, 1))
x_train = []
y_train = []
for i in range(10):
    x_train.append(img1)
    y_train.append(img2)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

# model = Sequential()
# model.add(Conv2D(3, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=(100,100,3), padding='same'))
# model.add(Conv2D(3, (3,3), activation='sigmoid',padding='same'))
# model.summary()



input_img = Input(shape=(100, 100,1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), padding='same')(x)

autoencoder_model = Model(input_img, decoded)

autoencoder_model.compile(loss = keras.losses.mse, optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
autoencoder_model.fit(x_train, y_train, batch_size=1, epochs=10000, verbose = 1)
autoencoder_model.save('lamborghini.h5')

model = load_model('lamborghini.h5')
result = model.predict(x_train)
print(result)



#
#
#
#
# model.compile(loss=keras.losses.mse,
#               optimizer=keras.optimizers.Adam(lr=0.01),
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=10000,
#           verbose=1)
# model.save('lamborghini.h5')
# result = model.predict(x_train)
#
#
#
# print(result)