import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, InputLayer
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_image(image_path, target_size=(256, 256)):
    image = load_img(image_path, color_mode='grayscale', target_size=target_size)
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Wczytanie plików
train_image_path = '/path_here'
test_image_path = '/path_here'
train_image = load_image(train_image_path)
test_image = load_image(test_image_path)

# Filtry Sobela
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32).reshape((3, 3, 1, 1))
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32).reshape((3, 3, 1, 1))

# Model z jedną konwolucją
model = Sequential([
    InputLayer(input_shape=(256, 256, 1)),
    Conv2D(1, (3, 3), padding='same', activation='linear', use_bias=False)
])
model.compile(optimizer=Adam(), loss=MeanSquaredError())
history = model.fit(train_image, train_image, epochs=10, verbose=0)

# Wykres błędu
plt.figure()
plt.plot(history.history['loss'])
plt.xlabel('Epoka')
plt.ylabel('MSE')
plt.title('Training Loss')
plt.show()

weights = model.layers[0].get_weights()[0]

plt.figure()
plt.title('Wagi Filtra')
plt.imshow(weights[:, :, 0, 0], cmap='gray')
plt.colorbar()
plt.show()

# Konwolucja naiwna
def naive_sobel_convolution(image, sobel_filter):
    return tf.nn.conv2d(image, sobel_filter, strides=[1, 1, 1, 1], padding='SAME')

naive_output_x = naive_sobel_convolution(test_image, sobel_x)

# Wykres
plt.figure()
plt.title('Konwolucja Naiwna - Wyjście')
plt.imshow(naive_output_x[0, :, :, 0], cmap='gray')
plt.colorbar()
plt.show()

# Dodatkowe zadanie: Sieć z dwoma konwolucjami i podwójną naiwną konwolucją
# Model z dwoma konwolucjami
model_two_conv = Sequential([
    InputLayer(input_shape=(256, 256, 1)),
    Conv2D(1, (3, 3), padding='same', activation='linear', use_bias=False, name='conv_x'),
    Conv2D(1, (3, 3), padding='same', activation='linear', use_bias=False, name='conv_y')
])
model_two_conv.compile(optimizer=Adam(), loss=MeanSquaredError())


history_two_conv = model_two_conv.fit(train_image, train_image, epochs=2000, verbose=0)

plt.figure()
plt.plot(history_two_conv.history['loss'])
plt.xlabel('Epoka')
plt.ylabel('MSE')
plt.title('Training Loss - 2 konwolucje')
plt.show()

# Wizualizacja wag konwolucji
weights_conv_x = model_two_conv.get_layer('conv_x').get_weights()[0]
weights_conv_y = model_two_conv.get_layer('conv_y').get_weights()[0]

plt.figure()
plt.title('Wagi Filtra Konwoucji X')
plt.imshow(weights_conv_x[:, :, 0, 0], cmap='gray')
plt.colorbar()
plt.show()

plt.figure()
plt.title('Wagi FIltra Konwolucji Y')
plt.imshow(weights_conv_y[:, :, 0, 0], cmap='gray')
plt.colorbar()
plt.show()

# Zastosowanie naiwnych konwolucji
naive_output_y = naive_sobel_convolution(test_image, sobel_y)
plt.figure()
plt.title('Konwolucja Naiwna - X')
plt.imshow(naive_output_x[0, :, :, 0], cmap='gray')
plt.colorbar()
plt.show()

plt.figure()
plt.title('Konwolucja Naiwna - Y')
plt.imshow(naive_output_y[0, :, :, 0], cmap='gray')
plt.colorbar()
plt.show()

# Porównanie wyników
plt.figure()
plt.subplot(1, 2, 1)
plt.title('Konwolucja Naiwna X - Wynik')
plt.imshow(naive_output_x[0, :, :, 0], cmap='gray')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title('Konwolucja Naiwna Y - Wynik')
plt.imshow(naive_output_y[0, :, :, 0], cmap='gray')
plt.colorbar()
plt.show()