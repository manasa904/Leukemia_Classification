# Leukemia_Classification
Leukemia Classification using ResNet50 <br>
Project Overview  <br>
This project develops an image classification model to detect Acute Lymphoblastic Leukemia (ALL) from microscopic blood smear images. It utilizes transfer learning with the ResNet50 convolutional neural network to classify images as either ALL (leukemia) or HEM (healthy).  <br>

Key Features  <br>
Automated Image Preprocessing: Includes steps like CLAHE for contrast enhancement, Otsu's thresholding for segmentation, smart cropping to focus on cells, and resizing. <br>

Transfer Learning: Leverages a pre-trained ResNet50 model (on ImageNet) as a powerful feature extractor. <br>

Deep Learning Model: A Keras sequential model built on ResNet50 with custom classification layers and dropout for regularization. <br>

Optimized Training: Uses Adam optimizer, binary_crossentropy loss, Early Stopping, and Model Checkpointing. <br>

Performance Evaluation: Provides test accuracy, classification report, and confusion matrix. <br>

Training Visualization: Plots model accuracy and loss history. <br>

Dataset <br>
The model is trained on the C-NMC Leukemia dataset, containing ALL and HEM blood smear images. <br>

Dataset Source: Kaggle - Leukemia Classification (C-NMC) <br>

Results <br>
The model effectively classifies leukemia images, achieving a test accuracy of approximately 99.91%. <br>

Test Loss: 0.0051 <br>
Test Accuracy: 0.9991 <br>

Classification Report: <br>

              precision    recall  f1-score   support

           0       1.00      0.99      1.00       678
           1       1.00      1.00      1.00      1455

    accuracy                           1.00      2133
   
Confusion Matrix: <br>

True Negatives (Correctly predicted HEM): 672 

False Positives (Predicted ALL, but was HEM): 6

False Negatives (Predicted HEM, but was ALL): 0

True Positives (Correctly predicted ALL): 1455


![Model confusion_matrix Plot](confusion_matrix_heatmap.png)

 ![Model Accuracy Plot](model_accuracy_plot.png)
 
 ![Model Loss Plot](model_loss_plot.png)
