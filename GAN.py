from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.layers import Reshape
from keras.layers import Activation
from keras.layers import LeakyReLU
from keras.activations import relu
from keras.activations import sigmoid
from keras.layers import BatchNormalization
from keras.layers import UpSampling2D
from keras.layers import Permute
from keras.layers import Convolution2D
from keras.layers import Flatten
from keras.optimizers import Adam
import numpy as np
import cv2
import os
import argparse
import matplotlib.pyplot as plt
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def generator_model(input_dim = 100, output_dim = 64, conv_activations = relu, output_activation = sigmoid):
    """Creates the generator model.

    Args:
      input_dim: Dimensionality of the input noise
      output_dim: Dimensionality of the output image, this implementation considers RGB square images
      conv_activations: Used activation after each of the applied convolutions
      output_activation: Activation used to predict the value of each of the output pixels, use sigmoid if image output between (0,1) tanh if (-1,1)

    Returns:
      The generator 

    """
    input_discriminatorim=  output_dim//2**4 ## 2^Number of convolutions blocks applied
    labels_discriminatorim = output_dim//2**4
    model = Sequential()
    model.add(Dense(512*input_discriminatorim*labels_discriminatorim, input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Activation(conv_activations))
    model.add(Reshape( (input_discriminatorim, labels_discriminatorim,512 ) ) )
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(256, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(conv_activations))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(128, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(conv_activations))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(64, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(conv_activations))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(3, (5, 5), padding='same',activation=output_activation))
    model.add(Permute((3, 1, 2)))
    return model

def discriminator_model(input_dim = (3, 64, 64), conv_activations = LeakyReLU(), output_activation = sigmoid):
    """Creates the discriminator model.

    Args:
      input_dim: Dimensionality of the input image
      conv_activations: Used activation after each of the applied convolutions
      output_activation: Activation used to predict if the image is real or false, use sigmoid if outputs between 0 or 1 tanh if outputs -1 or 1

    Returns:
      The discriminator 

    """
    model = Sequential()
    model.add(Convolution2D(64, (5, 5), strides=2, input_shape=input_dim, padding = 'same'))
    model.add(conv_activations)
    model.add(Dropout(0.2))
    model.add(Convolution2D(128, (5, 5), strides=2, padding = 'same'))
    model.add(conv_activations)
    model.add(Dropout(0.2))
    model.add(Convolution2D(256, (5, 5), strides=2, padding = 'same'))
    model.add(conv_activations)
    model.add(Dropout(0.2))
    model.add(Convolution2D(512, (5, 5), strides=2, padding = 'same'))
    model.add(conv_activations)
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1, activation=output_activation)) 
    return model


def generator_discriminator_model(generator, discriminator):
    """Creates the generator and discriminator model used in the step two of the training process where the generator is trained using the discriminator gradient

    Args:
      generator: The image generator model
      discriminator: The discriminator between real and sinthetic images

    Returns:
      The generator -> discriminator model
    """
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model


def load_image(path,image_dim = (64, 64)):
    """Load the image and resize it if needed

    Args:
      path: The path to the image
      image_dim: The expected dimensionality of the image in height and width

    Returns:
      The rescaled image
    """
    img = cv2.imread(path, 1)
    img = np.float32(cv2.resize(img, image_dim)) / 255 
    img = np.rollaxis(img, 2, 0)
    return img

def noise_vector(size = 100):
    """Create the noise vector which will be used as the input of the generator

    Args:
      size: The size of the noise vector created

    Returns:
      An array of random numbers for using as input of the generator
    """
    noise = np.random.uniform(-1, 1, size)
    return noise

def chunks(images, batch_size=16):
    """ Generator function that returns as many images as the size of the designed batch in each call

    Args:
      images: The original array of the training images
      batch_size: Size of each of the batches of images used to train the model
    Returns:
      A batch of real images for training purposes
    """
    for batch_initial_position in range(0, len(images), batch_size):
        yield images[batch_initial_position:batch_initial_position+batch_size]


def train(path, batch_size=16, EPOCHS=2000, epochs_between_saves = 20, epochs_between_outputs=10):
    """ Train the GAN and save the model each x epochs

    Args:
      path: The path to the original images used for training
      batch_size: Size of each of the batches of images used to train the model
      inter_model_margin: Maximun difference between the models needed to train with the next step of data
      EPOCHS: Number of epochs in which the model will be trained
      epochs_between_saves: How many epochs should the model train between saves
      epochs_between_output: How many epochs should the model train between saving a sample of the progresss

    """
    fig = plt.figure()
    paths = os.listdir(path)
    IMAGES = np.array( [ load_image(path + '/'+ img) for img in paths ] )
    np.random.shuffle( IMAGES )
    BATCHES = [ b for b in chunks(IMAGES, batch_size) ]

    print ("Number of batches", len(BATCHES))
    print ("Batch size is", batch_size)

    discriminator = discriminator_model()
    generator = generator_model()    
    discriminator_on_generator = generator_discriminator_model(generator, discriminator) 
    adam_gen=Adam(learning_rate=0.00002, beta_1=0.0005, beta_2=0.999, epsilon=1e-08)
    adam_dis=Adam(learning_rate=0.00002, beta_1=0.0005, beta_2=0.999, epsilon=1e-08)    
    generator.compile(loss='binary_crossentropy', optimizer=adam_gen)
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=adam_gen)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=adam_dis)

    for epoch in range(EPOCHS):
        print()
        print ("Epoch: ", epoch)
        print()
        time.sleep(1) 

        for index, image_batch in enumerate(BATCHES):
            Noise_batch = np.array( [ noise_vector() for n in range(len(image_batch)) ] )
            generated_images = generator.predict(Noise_batch)

            if epoch%epochs_between_outputs == 0 and index == 0: #Output some samples on how the model is progressing
                for i, img in enumerate(generated_images[:6]):
                    i = i+1
                    plt.subplot(2, 3, i)
                    img_rgb = np.rollaxis(img, 0, 3)
                    plt.imshow(img_rgb)
                    plt.axis('off')
                    fig.canvas.draw()
                    plt.savefig('results/Epoch_' + str(epoch) + '.png')
                    
            #Usual training process, train first discriminator on the batch, then train the generator on the same batch
            # Step 1: Generate Synthetic and concatenate them to real images, then train the discriminator on both
            discriminator.trainable = True
            input_discriminator = np.concatenate((image_batch, generated_images))
            labels_discriminator = np.array([1] * len(image_batch) + [0] * len(generated_images)) # labels
            discriminator_loss = discriminator.train_on_batch(input_discriminator, labels_discriminator)

            # Step 2: Generate random noise and use it to create train only the generator, we do this by freezing the discriminator weights and "lying" to the discriminator by telling it that the samples are real
            labels_generator = np.array([1] * len(image_batch))
            discriminator.trainable = False #Important to set the trainable of the discriminator to False so only forward propagations is applied to it and no learning process happens on this step
            generator_loss = discriminator_on_generator.train_on_batch(Noise_batch, labels_generator)

            print ("Initial batch losses : ", "Generator loss", generator_loss, "Discriminator loss", discriminator_loss, "Total:", generator_loss + discriminator_loss)
            if epoch % epochs_between_saves == 0 and epoch != 0:
                print ('Saving weights..')
                generator.save('models/generator_' + str(epoch) + '.h5', True)
                discriminator.save('models/discriminator_' + str(epoch) + '.h5', True)
        
    print ('Saving weights..')
    generator.save('models/generator_' + str(epoch) + '.h5', True)
    discriminator.save('models/discriminator_' + str(epoch) + '.h5', True)

path = 'CroppedYalePNG'
train(path,batch_size=32,EPOCHS=4000)