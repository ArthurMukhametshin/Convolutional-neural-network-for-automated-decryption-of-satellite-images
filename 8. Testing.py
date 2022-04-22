from Simple_multiclass_unet_model import multiclass_unet_model
from tensorflow.keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

n_classes = 6

train_images = []
for directory_path in glob.glob('C:/OSM/NeuralNetwork/CNN/Images/Training_set/Learning_dataset/'):
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR|cv2.IMREAD_ANYDEPTH)
        train_images.append(img)
train_images = np.array(train_images)

train_masks = []
for directory_path in glob.glob('C:/OSM/NeuralNetwork/CNN/Images/Training_set/Masks_dataset/'):
    for mask_path in glob.glob(os.path.join(directory_path, "*.tif")):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE|cv2.IMREAD_ANYDEPTH)
        train_masks.append(mask)
train_masks = np.array(train_masks)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
n, h, w = train_masks.shape # определяем параметры массива
train_masks_reshaped = train_masks.reshape(-1, 1) # раскладываем массив в один столбец
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped) # присваиваем метки для значений
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w) # возвращаем массив в исходное состояние
np.unique(train_masks_encoded_original_shape) # находим уникальные значения массива

train_images = normalize(train_images, axis=1) # нормализуем массив вдоль оси 1
train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3) # увеличиваем количество осей массива масок до 3

# Отбираем 10% на тестирование и оставшиеся на обучение
from sklearn.model_selection import train_test_split
X1, X_test, y1, y_test = train_test_split(train_images, train_masks_input, test_size=0.10, random_state=0) # X1 - изображения на обучение, X_test - изображения на тестирование (10%)
                                                                                                           # y1 - маски на обучение, y_test - маски на тестирование (10%)
# Дальнейшее разделение данных для обучения на меньшее подмножество для быстрого тестирования моделей
X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size=0.2, random_state=0) # X_train - отщипляем ещё 20% от тренировочного датасета изображений
                                                                                                       # y_train - отщипляем ещё 20% от тренировочного датасета масок
print('Параметры массивов: ', '1) датасета тренировочных изображений:', X_train.shape, '2) датасета тренировочных масок:', y_train.shape)
print('Параметры массивов: ', '1) датасета тестовых изображений:', X_test.shape, '2) датасета тестовых масок:', y_test.shape)
print('Значения классов в наборе данных: ', np.unique(y_train))  # получаем уникальные классы из массива масок для тренировки

from tensorflow.keras.utils import to_categorical

train_masks_cat = to_categorical(y_train, num_classes=n_classes) # создаем категории из массива масок для обучения
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes)) # переформировываем форму массива категорий в форму массива масок для обучения

test_masks_cat = to_categorical(y_test, num_classes=n_classes) # создаем категории из массива масок для тестирования
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes)) # переформировываем форму массива категорий в форму массива масок для тестирования

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(train_masks_reshaped_encoded),
                                                  train_masks_reshaped_encoded) # получаем веса для каждого из класса
print('Веса классов:', class_weights)

# Получаем параметры массива для обучения изображений
IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]
print ('Высота изображений для обучения: ', IMG_HEIGHT, 'Ширина изображений для обучения: ', IMG_WIDTH, ' Число каналов изображений для обучения: ', IMG_CHANNELS)

model = load_model('test.hdf5')
model.load_weights('test.hdf5')
model.summary()

import random
i = 1

while i < 21:
    test_img_number = random.randint(0, len(X_test))
    print ('1...')
    test_img = X_test[test_img_number]
    print ('2...')
    # print('Тестовое изображение: ', test_img)
    print('Размерность тестового изображения при загрузке:', test_img.shape)
    ground_truth = y_test[test_img_number]
    print ('3...')
    # test_img_norm = test_img[:, :, 0][:, :, None]
    # print ('4...')
    # print('Размерность тестового изображения после преобразования:', test_img_norm.shape)
    test_img_input = np.expand_dims(test_img, 0)
    print ('5...')
    print('Размерность тестового изображения после expand_dims:', test_img_input.shape)
    prediction = (model.predict(test_img_input))
    print ('6...')
    predicted_img = np.argmax(prediction, axis=3)[0, :, :]
    print ('7...')

    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:, :, 0], cmap='hsv')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:, :, 0], cmap='jet')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(predicted_img, cmap='jet')
    plt.legend()
    plt.show()
    i += 1
