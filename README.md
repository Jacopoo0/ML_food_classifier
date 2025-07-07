# Food Image Classifier - README


## Project Description

This project implements a convolutional neural network (CNN) to classify images of food into different categories. The system includes:

- Data exploration and visualization
- Image preprocessing pipeline
- CNN model training and evaluation
- Model testing and prediction visualization

## Files

### `main.py`
The main training script that:
1. Loads and explores the dataset
2. Preprocesses and splits the data (70% train, 15% validation, 15% test)
3. Defines and trains a CNN model
4. Evaluates model performance
5. Saves the trained model as `cibo_classifier.keras`

### `test_modello.py`
The testing script that:
1. Loads the trained model
2. Makes predictions on test set images
3. Visualizes predictions with true labels (green=correct, red=incorrect)

## Dataset Structure

The dataset should be organized in the following structure:
```
progetto/
└── progetto/
    └── dataset/
        ├── class1/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        ├── class2/
        │   ├── image1.jpg
        │   └── ...
        └── ...
```

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Matplotlib
- Seaborn
- Pandas
- NumPy

Install requirements with:
```bash
pip install tensorflow matplotlib seaborn pandas numpy
```

## Usage

1. **Training the model**:
```bash
python main.py
```

2. **Testing the model**:
```bash
python test_modello.py
```

## Model Architecture

The CNN model consists of:
- Input layer (256x256x3)
- Rescaling layer (normalizes pixel values)
- Two convolutional blocks (Conv2D + MaxPooling + Dropout)
- Flatten layer
- Dense output layer with softmax activation

## Results

The script generates several visualizations:
1. Heatmap of image counts per category
2. Sample images from the dataset
3. Training/validation accuracy plots
4. Prediction results on test images

## Author

[Jacopoo0](https://github.com/Jacopoo0)
