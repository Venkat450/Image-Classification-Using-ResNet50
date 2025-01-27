# Image Classification using ResNet50

## Overview
This project explores the application of transfer learning for image classification using the ResNet50 model. The dataset comprises images of rocks categorized into 30 classes. The project demonstrates steps like data preprocessing, augmentation, fine-tuning, and analysis of the model's performance. 

## Dataset
The dataset used for this project is available at [OSF Rock Dataset](https://osf.io/d6b9y/). It consists of:
- **360 Rocks Folder**: Used for training, containing 30 categories with 12 images per category.
- **120 Rocks Folder**: Used for validation, containing 30 categories with 4 images per category.

## Requirements
The following Python libraries are required:
- `tensorflow`
- `numpy`
- `pandas`
- `matplotlib`

Install dependencies using:
```bash
pip install tensorflow numpy pandas matplotlib
```
# Image Classification using ResNet50

## Project Workflow

### Dataset Preparation
- Grouped images into 30 categories by rock type.
- Created training and validation datasets using 12 and 4 images per category, respectively.

### Data Augmentation and Input Pipeline
- Applied augmentation techniques like rotation, shear, zoom, and flipping.
- Normalized pixel values to the range [0, 1].

### Model Architecture
- Used ResNet50 as the base model with ImageNet weights.
- Modified the architecture by removing the top layer and adding:
  - `GlobalAveragePooling2D`
  - Dense layers with 512 and 128 neurons
  - Dropout layers for regularization
  - Final Dense layer with 30 neurons and a softmax activation for classification.

### Training
- Initially trained only the newly added layers for a few epochs.
- Fine-tuned the entire model for 250 epochs.
- Utilized `EarlyStopping` and `ReduceLROnPlateau` callbacks for efficient training.

### Evaluation and Visualization
- Plotted training and validation accuracy and loss across epochs.
- Analyzed the performance trends and convergence.

### Correlation Analysis
- Calculated correlation coefficients for the next-to-last layer using Procrustes analysis to understand the relationship between the model's predictions and human judgment data.

## Results
- **Training Accuracy**: Gradually improved to **75.87%**.
- **Validation Accuracy**: Achieved **29.46%** after 250 epochs.

### Challenges
- Observed overfitting as validation accuracy plateaued while training accuracy continued to increase.
- Performance improved by tweaking hyperparameters, including batch size and dropout rates.

## How to Run
1. Clone the repository:
   ```bash
   git clone <repository-link>
   cd <repository-folder>
2. Organize the dataset directories as specified in the code.
3. Run the provided Jupyter Notebook or script to train and evaluate the model.

## Discussion

### Network Performance
- ResNet50 showed consistent improvements in training accuracy; however, validation accuracy remained low, indicating potential overfitting.

### Convergence
- Training loss steadily decreased and stabilized, but validation loss showed fluctuations, pointing to challenges in generalization.

### Hyperparameter Tweaks
- Adjustments in batch size led to better training consistency.
- Including dropout and batch normalization layers effectively reduced overfitting.

## Future Improvements
- Test different architectures like EfficientNet for potentially better generalization.
- Experiment with enhanced data augmentation techniques and dynamic learning rate schedules.
- Apply additional regularization methods to further mitigate overfitting.

## Acknowledgments
Gratitude to the creators of the [OSF Rock Dataset](https://osf.io/d6b9y/) and the TensorFlow and Keras communities for providing robust tools to build and train deep learning models.
