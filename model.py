from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt


def build_data_pipeline(train_directory, validation_directory):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False
    )
    validation_datagen = ImageDataGenerator(
        rescale=1./255
    )
    train_generator = train_datagen.flow_from_directory(
        train_directory,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )
    validation_generator = validation_datagen.flow_from_directory(
        validation_directory,
        target_size=(150, 150),
        batch_size=64,
        class_mode='categorical'
    )
    return train_generator, validation_generator


def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
                     input_shape=(150, 150, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['acc']
    )
    return model


def train_model(model, train_generator, validation_generator):
    epochs = 10
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping, ]
    )
    return history


def plot_training_validation_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend(True)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend(True)
    plt.show()


def model_save(model):
    model.save('model.h5')


def model_load():
    model_filename = 'model.h5'
    return load_model(model_filename)


def make_predictions(model, data):
    return model.predict(data)


if __name__ == '__main__':
    train_directory = 'final/train'
    validation_directory = 'final/validation'
    train_generator, validation_generator = build_data_pipeline(
        train_directory,
        validation_directory
    )
    model = build_model()
    history = train_model(model, train_generator, validation_generator)
    model_save(model)
