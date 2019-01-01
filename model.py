import os
import csv
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten , Cropping2D 
from sklearn.utils import shuffle

samples = []
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # Skip first header
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
import cv2
import numpy as np
import sklearn
#Generator function to access data on-demand
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            steering = []
            for batch_sample in batch_samples:
                name = '/opt/carnd_p3/data/IMG/'+batch_sample[0].split('/')[-1]
                #center image
                center_image = cv2.imread(name)
                # convert center image to rgb
                cntr_image_rgb = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                # append center rgb images
                images.append(cntr_image_rgb)
                center_angle = float(batch_sample[3])
                # steering angle
                steering.append(center_angle)
                # flip the center rgb image
                images.append(cv2.flip(cntr_image_rgb, 1))
                steering.append(-center_angle)
                
                # left image
                name = '/opt/carnd_p3/data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                # convert left image to RGB
                left_image_rgb = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                # append left RGB images
                images.append(left_image_rgb)
                left_angle = float(batch_sample[3]) + 0.1
                steering.append(left_angle)
                # append flipped left rgb images
                images.append(cv2.flip(left_image_rgb, 1))
                steering.append(-left_angle)
                
                # right image
                name = '/opt/carnd_p3/data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                # convert to right images to RGB
                right_image_rgb = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                # append right RGB
                images.append(right_image_rgb)
                right_angle = float(batch_sample[3]) - 0.1
                steering.append(right_angle)
                # append flipped right rgb
                images.append(cv2.flip(right_image_rgb, 1))
                steering.append(-right_angle)
            
            X_train = np.array(images)
            y_train = np.array(steering)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()

model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
#Convolution Layer 1 
model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu'))
#Dropout after Layer 1
model.add(Dropout(.25))
#Convolution of Layer 2
model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
#Droput of Layer after layer 2
model.add(Dropout(.05))
#Implementing MaxPooling to feed forward only the maximum stride pixels
model.add(MaxPooling2D((1, 1), border_mode='valid'))
model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
#Output for steering
model.add(Conv2D(64, (3,3), activation='relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch= len(train_samples) // 32,
validation_data=validation_generator, validation_steps=len(validation_samples) // 32, epochs=5, verbose = 1)

model.save('model.h5')