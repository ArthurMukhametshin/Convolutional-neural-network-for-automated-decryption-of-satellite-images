from Simple_multiclass_unet_model import multiclass_unet_model

from tensorflow.keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt

n_classes = 6  # Количество классов для сегментации

# Представляем информацию о тренировочных изображениях в виде списка
train_images = []

for directory_path in glob.glob('C:/OSM/NeuralNetwork/CNN/Images/Training_set2/Learning_dataset/'):
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR|cv2.IMREAD_ANYDEPTH)
        train_images.append(img)

# Конвертируем список в массив для обучения
train_images = np.array(train_images)

# Представляем информацию о масках в виде списка
train_masks = []
for directory_path in glob.glob('C:/OSM/NeuralNetwork/CNN/Images/Training_set2/Masks_dataset/'):
    for mask_path in glob.glob(os.path.join(directory_path, "*.tif")):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE|cv2.IMREAD_ANYDEPTH)
        train_masks.append(mask)

# Конвертируем список в массив для обучения
train_masks = np.array(train_masks)

###############################################
# Начинаем кодировать метки, имея дело с многомерным массивом, поэтому необходимо его сгладить, кодировать и изменить форму
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
n, h, w = train_masks.shape # определяем параметры массива
train_masks_reshaped = train_masks.reshape(-1, 1) # раскладываем массив в один столбец
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped) # присваиваем метки для значений
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w) # возвращаем массив в исходное состояние
np.unique(train_masks_encoded_original_shape) # находим уникальные значения массива

#################################################
train_images = normalize(train_images, axis=1) # нормализуем массив вдоль оси 1

train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3) # увеличиваем количество осей массива масок до 3

# Создаем подмножество данных для быстрого тестирования
# Отбираем 10% на тестирование и оставшиеся на обучение
from sklearn.model_selection import train_test_split

X1, X_test, y1, y_test = train_test_split(train_images, train_masks_input, test_size=0.10, random_state=0)

# Дальнейшее разделение данных для обучения на меньшее подмножество для быстрого тестирования моделей
X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size=0.2, random_state=0)

print('Значения классов в наборе данных: ', np.unique(y_train))  # 0 это фон

from tensorflow.keras.utils import to_categorical

train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

###############################################################
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(train_masks_reshaped_encoded),
                                                  train_masks_reshaped_encoded)
print('Веса классов:', class_weights)

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]


def get_model():
    return multiclass_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)


model = get_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train_cat,
                    batch_size=16,
                    verbose=1,
                    epochs=50,
                    validation_data=(X_test, y_test_cat),
#                    class_weight=class_weights,
                    shuffle=False)

model.save('test_2.hdf5')
############################################################
# Оцениваем модель
_, acc = model.evaluate(X_test, y_test_cat)
print("Точность = ", (acc * 100.0), "%")

###
# Строим график обучения и проверки точности и потерь для каждой эпохи
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

##################################
# model = get_model()
model.load_weights('test.hdf5')
# model.load_weights('sandstone_50_epochs_catXentropy_acc_with_weights.hdf5')

# IOU
y_pred = model.predict(X_test)
y_pred_argmax = np.argmax(y_pred, axis=3)

##################################################

# Используем встроенную функцию keras
from keras.metrics import MeanIoU

n_classes = 6
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test[:, :, :, 0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

# Подсчитываем I0U для каждого класса
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0, 0] / (
            values[0, 0] + values[0, 1] + values[0, 2] + values[0, 3] + values[0, 4] + values[0, 5] + values[1, 0] + values[2, 0] + values[3, 0] + values[4, 0] + values[5, 0])
class2_IoU = values[1, 1] / (
            values[1, 1] + values[1, 0] + values[1, 2] + values[1, 3] + values[1, 4] + values[1, 5] + values[0, 1] + values[2, 1] + values[3, 1] + values[4, 1] + values[5, 1])
class3_IoU = values[2, 2] / (
            values[2, 2] + values[2, 0] + values[2, 1] + values[2, 3]  + values[2, 4] + values[2, 5] + values[0, 2] + values[1, 2] + values[3, 2] + values[4, 2] + values[5, 2])
class4_IoU = values[3, 3] / (
            values[3, 3] + values[3, 0] + values[3, 1] + values[3, 2] + values[3, 4] + values[3, 5] + values[0, 3] + values[1, 3] + values[2, 3] + values[4, 3] + values[5, 3])
class5_IoU = values[4, 4] / (
            values[4, 4] + values[4, 0] + values[4, 1] + values[4, 2] + values[4, 3] + values[4, 5] + values[0, 4] + values[1, 4] + values[2, 4] + values[3, 4] + values[5, 4])
class6_IoU = values[5, 5] / (
            values[5, 5] + values[5, 0] + values[5, 1] + values[5, 2] + values[5, 3] + values[5, 4] + values[0, 5] + values[1, 5] + values[2, 5] + values[3, 5] + values[4, 5])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)
print("IoU for class5 is: ", class5_IoU)
print("IoU for class6 is: ", class6_IoU)
