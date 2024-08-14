import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from keras.utils import load_img
from keras.utils import img_to_array
import cv2


# Assigning file paths to variables
train_dir = 'C:/Users/ozkan/Desktop/FacialExpressionRecognition/train'
test_dir = 'C:/Users/ozkan/Desktop/FacialExpressionRecognition/test'
model_path = 'C:/Users/ozkan/Desktop/deep/FacialExpressionRecognition.h5'




# Data augmentation and ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    class_mode='categorical'
)



# Model
def create_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(7, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


model = create_model()

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=4, verbose=1)

# Training
history = model.fit(
    training_set,
    epochs=20,
    batch_size=32,
    validation_data=test_set,
    shuffle=True,
    callbacks=[early_stop]
)


# Model saving
model.save(model_path)


# Model loading
model = load_model(model_path)

# Visualizing
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(history.history['accuracy'], label='Training Accuracy')
ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
ax[0].set_title('Training and Validation Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Accuracy')
ax[0].legend()

ax[1].plot(history.history['loss'], label='Training Loss')
ax[1].plot(history.history['val_loss'], label='Validation Loss')
ax[1].set_title('Training and Validation Loss')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].legend()

plt.tight_layout()
plt.show()



# Class Distrubition in the Dataset
def plot_class_distribution(generator):
    labels = list(generator.class_indices.keys())
    counts = list(generator.classes)
    class_counts = pd.Series(counts).value_counts().sort_index()
    class_counts.index = labels

    plt.figure(figsize=(10, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
    plt.title('Class Distribution in the Dataset')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

plot_class_distribution(training_set)



#Confision matrix Heatmap
def plot_heatmap_confusion_matrix(model, generator):
    # Modelin tahminlerini al
    Y_pred = model.predict(generator)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = generator.classes

    # Karışıklık matrisini hesapla
    cm = confusion_matrix(y_true, y_pred)

    # Sınıf adlarını al
    class_names = list(generator.class_indices.keys())

    # Karışıklık matrisini görselleştir
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix Heatmap')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()

plot_heatmap_confusion_matrix(model, test_set)




# Printing accuracies
train_loss, train_acc = model.evaluate(training_set)
test_loss, test_acc   = model.evaluate(test_set)
print("Final train accuracy = {:.2f} , validation accuracy = {:.2f}".format(train_acc*100, test_acc*100))




# Entering a certain photo and getting results

img = load_img("C:/Users/ozkan/Desktop//FacialExpressionRecognition/test/angry/PrivateTest_731447.jpg",target_size = (48,48),color_mode = "grayscale")
img = np.array(img)
plt.imshow(img)
print(img.shape)


training_set.class_indices
label_dict = ['angry', 'disgust' , 'fear','happy', 'neutral', 'sad', 'surprise']

test_image = img_to_array(img)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
result[0]

res = np.argmax(result[0])
print('predicted Label for that image is: {}'.format(label_dict[res]))




# Initialize the camera
camera = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to match the model's expected sizing
    resized = cv2.resize(gray, (48, 48))
    
    # Preprocess the image
    image = img_to_array(resized)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    
    # Predict the emotion
    result = model.predict(image)
    
    # Get the predicted label
    label_dict = ['angry', 'disgust' , 'fear', 'happy', 'neutral', 'sad', 'surprise']
    res = np.argmax(result[0])
    predicted_label = label_dict[res]
    
    # Display the frame with the predicted label
    cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Facial Expression Recognition', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()
