import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical

# initial learning rate and number of epochs
LR = 0.0001
EPOCHS = 20

DIRECTORY = r"C:\Users\Rishab\IPCV_Project\dataset"
CATEGORIES = ["with_mask", "without_mask"]

# load the dataset of images into the list "data" and their respective labels(with mask, without mask)
# into the list "labels"
data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)

    	data.append(image)
    	labels.append(category)

# Convert the labels(with_mask, without_mask) into numeric data using one-hot encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#convert the lists to numpy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

#split the processed dataset into train and test datasets
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.25, stratify=labels, random_state=42)

# construct the training image generator which creates new images with slightly altered features(rotation,zoom)
# from the orignal image.
aug = ImageDataGenerator( rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# load the MobileNetV2 network(only the Conv layers excluding the FC layers) pre-trained on the "ImageNet" dataset
baseModel = MobileNetV2(weights="imagenet", input_tensor=Input(shape=(224, 224, 3)), include_top=False)

# construct the head of the model that will be placed on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# placing the fully connected model on top of the base model to create the full model
model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
	layer.trainable = False

# compiling the model
opt = Adam(lr=LR, decay=LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# training the network
H = model.fit( aug.flow(trainX, trainY, batch_size=30),
	steps_per_epoch=len(trainX) // 30,
	validation_data=(testX, testY),
	validation_steps=len(testX) // 30,
	epochs=EPOCHS)

# evaluating the network on the test dataset
pred = model.predict(testX, batch_size=30)

# for each image in the testing dataset we need to find the index of the
# label with corresponding largest predicted probability
pred = np.argmax(pred, axis=1)

print(classification_report(testY.argmax(axis=1), pred, target_names=lb.classes_))

# saving the model to disk
model.save("mask_detector.model", save_format="h5")

# plot the training and testing loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")