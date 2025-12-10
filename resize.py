import os
import random
import shutil
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# augmentation
datagen = ImageDataGenerator(
    rotation_range=30,       # Xoay ngẫu nhiên trong khoảng 30 độ
    width_shift_range=0.2,   # Dịch chuyển ngang
    height_shift_range=0.2,  # Dịch chuyển dọc
    shear_range=0.2,         # Biến dạng
    zoom_range=0.2,          # Zoom
    horizontal_flip=True,    # Lật ngang
    fill_mode='nearest'      # Điền pixel bị mất
)

# Thư mục chứa ảnh gốc
input_folder = r"D:\Code\Deep_Learning\Dataset\flower_img"
output_folder = "D:/Code/Deep_Learning/Dataset_224"

# Tạo thư mục train và test
train_folder = os.path.join(output_folder, "train")
test_folder = os.path.join(output_folder, "test")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Lặp qua từng loại hoa (subfolder)
for flower_class in os.listdir(input_folder):
    class_path = os.path.join(input_folder, flower_class)
    if not os.path.isdir(class_path):
        continue

    # Tạo thư mục cho train/test
    train_class_folder = os.path.join(train_folder, flower_class)
    test_class_folder = os.path.join(test_folder, flower_class)
    os.makedirs(train_class_folder, exist_ok=True)
    os.makedirs(test_class_folder, exist_ok=True)

    # Lấy danh sách ảnh và shuffle
    images = os.listdir(class_path)
    random.shuffle(images)

    # Chia thành 80% train, 20% test
    split_idx = int(0.8 * len(images))
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    def process_and_save(img_list, dest_folder):
        for img_name in img_list:
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path).convert("RGB")  # Load ảnh RGB
            img = img.resize((224, 224))  # Resize về 224x224
            img_array = np.array(img)

            # Augment ảnh
            img_array = np.expand_dims(img_array, axis=0)  # Thêm batch dimension
            augmented_images = datagen.flow(img_array, batch_size=1)

            # Lưu ảnh gốc
            img.save(os.path.join(dest_folder, img_name))

            # Lưu thêm ảnh augment
            for i in range(2):  # Tạo 2 ảnh augment mỗi ảnh gốc
                aug_img = next(augmented_images)[0].astype("uint8")
                aug_filename = f"aug_{i}_{img_name}"
                Image.fromarray(aug_img).save(os.path.join(dest_folder, aug_filename))

    # Xử lý train và test
    process_and_save(train_images, train_class_folder)
    process_and_save(test_images, test_class_folder)

print("✅ Hoàn tất resize, augmentation và chia tập train/test!")
