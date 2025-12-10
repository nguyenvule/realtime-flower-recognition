import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

save_dir = "results"
os.makedirs(save_dir, exist_ok=True)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f'✅ Đang sử dụng GPU: {gpus[0]}')
else:
    print("⚠️ Không tìm thấy GPU, chạy trên CPU!")

# Load mô hình đã train
model_path = "flower_cnn_224_model.h5"
transfer_model = load_model(model_path)
print(f"📥 Loaded model from {model_path}")

# Mở khóa 12 lớp cuối để fine-tune
for layer in transfer_model.layers[-12:]:
    layer.trainable = True

# Bên dịch lại mô hình với learning rate nhỏ hơn
transfer_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#Load dữ liệu (giữ nguyên dataset cũ)
train_directory = r'D:\Code\Deep_Learning\Dataset_224\train'
test_directory = r'D:\Code\Deep_Learning\Dataset_224\test'

train_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_directory, target_size=(224, 224), batch_size=32, class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_directory, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False)

# Tiếp tục fine-tune mô hình
history_finetune = transfer_model.fit(
    train_generator,  
    epochs=10,  # Số epoch fine-tune
    validation_data=test_generator
)

# Lưu lại mô hình fine-tuned
transfer_model.save(os.path.join(save_dir, "flower_finetuned_model.h5"))
print("📦 Fine-tuned model saved!")

# Vẽ biểu đồ Loss và Accuracy
def plot_history(history, save_path):
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Fine-tuned Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Fine-tuned Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')
    
    plt.savefig(save_path)  
    plt.show()

plot_history(history_finetune, os.path.join(save_dir, 'training_history_finetune.png'))

# Đánh giá mô hình sau fine-tune
test_loss, test_accuracy = transfer_model.evaluate(test_generator)
print(f"Final Test Accuracy after Fine-Tuning: {test_accuracy:.4f}")

# Vẽ Confusion Matrix
y_true = test_generator.classes
y_pred = transfer_model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
labels = list(test_generator.class_indices.keys())

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

plt.savefig(os.path.join(save_dir, 'confusion_matrix_finetune.png'))
plt.show()
