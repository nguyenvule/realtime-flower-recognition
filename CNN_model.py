import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Kiểm tra GPU và thiết lập bộ nhớ động
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f'✅ Đang sử dụng GPU: {gpus[0]}')
else:
    print("⚠️ Không tìm thấy GPU, chạy trên CPU!")

# Bật Mixed Precision để tăng tốc trên GPU
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax', dtype='float32')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig('training_history_cnn_224.png')
    plt.show()

def evaluate_model(model, test_generator):
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f'✅ Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}')
    
    y_true = test_generator.classes
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_generator.class_indices.keys())
    disp.plot()
    plt.savefig('confusion_matrix_cnn_224.png')
    plt.show()

def main():
    train_directory = r'D:\Code\Deep_Learning\Dataset_224\train'
    test_directory = r'D:\Code\Deep_Learning\Dataset_224\test'

    train_datagen = ImageDataGenerator(rescale=1.0/255)
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator = train_datagen.flow_from_directory(
        train_directory, target_size=(224, 224), batch_size=32, class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
        test_directory, target_size=(224, 224), batch_size=32, class_mode='categorical')

    # In ra danh sách class và ID tương ứng
    class_indices = train_generator.class_indices
    class_labels = {v: k for k, v in class_indices.items()}
    print("Danh sách class và ID tương ứng:")
    for class_id, class_name in class_labels.items():
        print(f"  - ID {class_id}: {class_name}")

    model = build_model(input_shape=(224, 224, 3), num_classes=len(class_indices))

    if os.path.exists('flower_cnn_224_model.h5'):
        model.load_weights('flower_cnn_224_model.h5')
        print("📥 Model loaded from flower_cnn_224_model.h5")
    else:
        print("🚀 Training the model on GPU...")
        history = model.fit(train_generator, epochs=20, validation_data=test_generator)
        model.save('flower_cnn_224_model.h5')
        print("📦 Model saved as flower_cnn_224_model.h5")
        plot_history(history)

    evaluate_model(model, test_generator)

if __name__ == "__main__":
    main()
