import sys
import cv2
import numpy as np
import tensorflow as tf
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt

MODEL_PATH = r"D:\Code\Deep_Learning\Final_Project\Code\results_finetunning\flower_finetuned_model.h5"#Thay đường dẫn
model = tf.keras.models.load_model(MODEL_PATH)
class_labels = ["Daisy", "Đanelion", "Rose", "Sunlower", "Tulip"]

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0 
    img = np.expand_dims(img, axis=0) 
    return img

def predict_image(img):
    img = preprocess_image(img)
    preds = model.predict(img)
    class_idx = np.argmax(preds)
    return class_labels[class_idx], preds[0][class_idx]

class FlowerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flower Classification")
        self.setGeometry(100, 100, 400, 500)
        
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.result_label = QLabel("Dự đoán: Chưa có ảnh", self)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.upload_button = QPushButton("Chọn ảnh từ máy", self)
        self.upload_button.clicked.connect(self.load_image)
        
        self.webcam_button = QPushButton("Chụp ảnh từ Webcam", self)
        self.webcam_button.clicked.connect(self.capture_webcam)
        
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.upload_button)
        layout.addWidget(self.webcam_button)
        self.setLayout(layout)
    
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            img = cv2.imread(file_path)
            self.display_and_predict(img)
    
    def capture_webcam(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            self.result_label.setText("Không thể mở webcam!")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                self.result_label.setText("Lỗi khi đọc ảnh từ webcam!")
                break
            cv2.imshow("Webcam - Nhấn Enter để chụp, ESC để thoát", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Phím Enter
                self.display_and_predict(frame)
                break
            elif key == 27:  # Phím ESC để thoát
                break

        cap.release()
        cv2.destroyAllWindows()

    def display_and_predict(self, img):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio))
        
        label, confidence = predict_image(img)
        self.result_label.setText(f"Dự đoán: {label} ({confidence:.2f})")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FlowerApp()
    window.show()
    sys.exit(app.exec())
