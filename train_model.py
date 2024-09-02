import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Reshape, UpSampling2D, LeakyReLU, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Configurar el directorio de datos
data_dir = '/users/jorge/sites/2024_JHON_BOY_MODEL/venv/dataset'  # Cambia esta ruta a la ubicación de tus imágenes

# Configurar el generador de datos
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = datagen.flow_from_directory(data_dir, target_size=(128, 128), color_mode='rgb', batch_size=32, subset='training')

def build_generator():
    model = Sequential([
        Input(shape=(100,)),  # Ajustar la entrada aquí
        Dense(128 * 16 * 16, activation="relu"),
        Reshape((16, 16, 128)),
        UpSampling2D(),
        Conv2D(128, kernel_size=3, padding="same"),
        LeakyReLU(negative_slope=0.2),  # Cambiar alpha por negative_slope
        UpSampling2D(),
        Conv2D(64, kernel_size=3, padding="same"),
        LeakyReLU(negative_slope=0.2),  # Cambiar alpha por negative_slope
        UpSampling2D(),
        Conv2D(3, kernel_size=3, padding="same", activation='tanh')
    ])
    return model

def build_discriminator():
    model = Sequential([
        Input(shape=(128, 128, 3)),
        Conv2D(64, kernel_size=3, strides=2, padding="same"),
        LeakyReLU(negative_slope=0.2),  # Cambiar alpha por negative_slope
        Dropout(0.25),
        Conv2D(128, kernel_size=3, strides=2, padding="same"),
        LeakyReLU(negative_slope=0.2),  # Cambiar alpha por negative_slope
        Dropout(0.25),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return model

# Construir y compilar los modelos
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

generator = build_generator()
generator_optimizer = Adam(0.0002, 0.5)

# Pérdida binaria cruzada
bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# Función de pérdida del discriminador
def discriminator_loss(real_output, fake_output):
    real_loss = bce(tf.ones_like(real_output), real_output)
    fake_loss = bce(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Función de pérdida del generador
def generator_loss(fake_output):
    return bce(tf.ones_like(fake_output), fake_output)

# Función de entrenamiento
@tf.function
def train_step(images):
    noise = tf.random.normal([images.shape[0], 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator.optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

def train(epochs, batch_size=32):
    print("Comenzando entrenamiento...")
    for epoch in range(epochs):
        print(f"Época {epoch+1}/{epochs}")
        for step, (real_images, _) in enumerate(train_data):
            gen_loss, disc_loss = train_step(real_images)
        if epoch % 10 == 0:
            generator.save(f'generator_model_epoch_{epoch}.keras', save_format='keras')
            print(f"Guardado modelo del generador en la época {epoch}")
        print(f"Época {epoch} [D loss: {disc_loss:.4f}] [G loss: {gen_loss:.4f}]")
    print("Entrenamiento completado")
    generator.save('generator_model_final.keras', save_format='keras')

train(epochs=20, batch_size=32)
