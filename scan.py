import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton,
    QFileDialog, QWidget, QListWidget, QMessageBox, QProgressBar, QSlider, QHBoxLayout,
    QComboBox, QButtonGroup, QRadioButton, QGroupBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
from fpdf import FPDF
import numpy as np

class DocumentScannerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Document Scanner")
        self.setGeometry(100, 100, 800, 900)  # Made taller for new controls
        self.denoise_strength = 10
        self.current_image = None
        self.original_gray = None
        self.processed_images = []  # Store multiple processed images for PDF
        self.denoise_methods = {
            "Non-Local Means": self.nl_means_denoise,
            "Gaussian": self.gaussian_denoise,
            "Median": self.median_denoise,
            "Bilateral": self.bilateral_denoise,
            "None": self.no_denoise
        }
        self.current_denoise_method = "Non-Local Means"
        self.initUI()

    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()

        # Upload section
        upload_layout = QHBoxLayout()
        self.upload_button = QPushButton("Upload Single Image")
        self.upload_button.clicked.connect(self.upload_image)
        self.upload_multiple_button = QPushButton("Upload Multiple Images")
        self.upload_multiple_button.clicked.connect(self.upload_multiple_images)
        upload_layout.addWidget(self.upload_button)
        upload_layout.addWidget(self.upload_multiple_button)
        self.layout.addLayout(upload_layout)

        # Image list
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.preview_selected_image)
        self.layout.addWidget(self.file_list)

        # Preview label
        self.preview_label = QLabel("Preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(600, 600)
        self.layout.addWidget(self.preview_label)

        # Denoise method selection
        self.denoise_method_layout = QHBoxLayout()
        self.denoise_method_label = QLabel("Denoise Method:")
        self.denoise_method_combo = QComboBox()
        self.denoise_method_combo.addItems(self.denoise_methods.keys())
        self.denoise_method_combo.currentTextChanged.connect(self.change_denoise_method)
        self.denoise_method_layout.addWidget(self.denoise_method_label)
        self.denoise_method_layout.addWidget(self.denoise_method_combo)
        self.layout.addLayout(self.denoise_method_layout)

        # Denoise strength control
        self.denoise_slider_layout = QHBoxLayout()
        self.denoise_label = QLabel("Denoise Strength: 10")
        self.denoise_slider = QSlider(Qt.Horizontal)
        self.denoise_slider.setMinimum(0)
        self.denoise_slider.setMaximum(30)
        self.denoise_slider.setValue(10)
        self.denoise_slider.valueChanged.connect(self.update_denoise_live)
        self.denoise_slider_layout.addWidget(self.denoise_label)
        self.denoise_slider_layout.addWidget(self.denoise_slider)
        self.layout.addLayout(self.denoise_slider_layout)

        # Save options group
        save_group = QGroupBox("Save Options")
        save_layout = QHBoxLayout()
        
        # Save buttons
        self.save_image_button = QPushButton("Save as Image")
        self.save_image_button.clicked.connect(self.save_image)
        self.save_image_button.setEnabled(False)
        
        self.save_pdf_button = QPushButton("Save as PDF")
        self.save_pdf_button.clicked.connect(self.save_pdf)
        self.save_pdf_button.setEnabled(False)
        
        save_layout.addWidget(self.save_image_button)
        save_layout.addWidget(self.save_pdf_button)
        save_group.setLayout(save_layout)
        self.layout.addWidget(save_group)

        self.central_widget.setLayout(self.layout)

    def upload_multiple_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Image Files", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        if files:
            self.file_list.clear()
            self.processed_images = []
            for file_path in files:
                self.file_list.addItem(os.path.basename(file_path))
                image = cv2.imread(file_path, cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                self.processed_images.append((file_path, gray))
            
            # Show first image
            self.original_gray = self.processed_images[0][1]
            self.update_denoise_live()
            self.save_image_button.setEnabled(True)
            self.save_pdf_button.setEnabled(True)

    def preview_selected_image(self, item):
        index = self.file_list.row(item)
        if index < len(self.processed_images):
            self.original_gray = self.processed_images[index][1]
            self.update_denoise_live()

    def save_pdf(self):
        if not self.processed_images:
            QMessageBox.warning(self, "No Images", "Please upload images first.")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF", "scanned_documents.pdf",
            "PDF Files (*.pdf)"
        )

        if save_path:
            pdf = FPDF()
            pdf.set_auto_page_break(0)

            for _, gray_image in self.processed_images:
                # Process image with current settings
                processed = self.process_image(gray_image, self.denoise_strength)
                
                # Save temporary image
                temp_path = "temp_img.png"
                cv2.imwrite(temp_path, processed)

                # Add to PDF
                pdf.add_page()
                page_width = pdf.w - 20
                page_height = pdf.h - 20
                pdf.image(temp_path, 10, 10, page_width, page_height)

            # Save PDF and clean up
            pdf.output(save_path)
            if os.path.exists("temp_img.png"):
                os.remove("temp_img.png")

            QMessageBox.information(self, "Success", "PDF saved successfully!")

    def save_image(self):
        if self.current_image is None:
            QMessageBox.warning(self, "No Image", "Please process an image first.")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "processed_image.png",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )

        if save_path:
            cv2.imwrite(save_path, self.current_image)
            QMessageBox.information(self, "Success", "Image saved successfully!")

    # [Previous methods remain the same: nl_means_denoise, gaussian_denoise, etc.]
    def nl_means_denoise(self, image, strength):
        return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)

    def gaussian_denoise(self, image, strength):
        kernel_size = 2 * round(strength / 5) + 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def median_denoise(self, image, strength):
        kernel_size = 2 * round(strength / 5) + 1
        return cv2.medianBlur(image, kernel_size)

    def bilateral_denoise(self, image, strength):
        d = int(strength / 2) + 1
        sigma_color = strength * 2
        sigma_space = strength
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    def no_denoise(self, image, strength):
        return image

    def change_denoise_method(self, method):
        self.current_denoise_method = method
        self.update_denoise_live()

    def convert_cv_to_pixmap(self, cv_img):
        height, width = cv_img.shape
        bytes_per_line = width
        q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        return QPixmap.fromImage(q_img)

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image File", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )

        if file_path:
            self.file_list.clear()
            self.processed_images = []
            self.file_list.addItem(os.path.basename(file_path))
            
            image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            self.original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.processed_images.append((file_path, self.original_gray))
            
            self.update_denoise_live()
            self.save_image_button.setEnabled(True)
            self.save_pdf_button.setEnabled(True)

    def process_image(self, gray_image, denoise_strength):
        denoised = self.denoise_methods[self.current_denoise_method](gray_image, denoise_strength)
        processed = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(processed)
        return enhanced

    def update_denoise_live(self):
        if self.original_gray is not None:
            self.denoise_strength = self.denoise_slider.value()
            self.denoise_label.setText(f"Denoise Strength: {self.denoise_strength}")

            processed = self.process_image(self.original_gray, self.denoise_strength)
            self.current_image = processed

            pixmap = self.convert_cv_to_pixmap(processed)
            scaled_pixmap = pixmap.scaled(
                600, 600,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled_pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DocumentScannerApp()
    window.show()
    sys.exit(app.exec_())