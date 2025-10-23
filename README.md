# Diabetes Prediction with Keras

A simple neural network built with Keras/TensorFlow to predict diabetes onset based on the Pima Indians Diabetes dataset. This project demonstrates the fundamentals of building, training, and evaluating a binary classification model.

## Description

This project uses a Sequential neural network to predict whether a patient has diabetes based on 8 medical predictor variables. The model is trained on the famous Pima Indians Diabetes dataset and achieves a straightforward binary classification task.

## Key Features

* **Binary Classification** - Predicts diabetes presence (yes/no)
* **Simple Neural Network** - 3-layer Sequential model architecture
* **Medical Data Analysis** - Uses real-world health metrics
* **Easy to Understand** - Perfect for beginners learning Keras
* **Model Evaluation** - Includes accuracy scoring and predictions

## Technologies Used

* **Python** 3.7+
* **TensorFlow** 2.x
* **Keras** (integrated with TensorFlow)
* **NumPy** - For data handling

## Installation Requirements

### Prerequisites

Make sure you have Python 3.7 or higher installed on your system.

### Install Dependencies

```bash
pip install tensorflow numpy
```

Or install from a requirements file:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
tensorflow>=2.0.0
numpy>=1.19.0
```

## Dataset

The project uses the **Pima Indians Diabetes Database**:
- **Source**: Originally from the National Institute of Diabetes and Digestive and Kidney Diseases
- **Samples**: 768 patients
- **Features**: 8 medical predictor variables
- **Target**: Binary outcome (0 = no diabetes, 1 = diabetes)

### Features (Input Variables):
1. Number of pregnancies
2. Plasma glucose concentration
3. Diastolic blood pressure (mm Hg)
4. Triceps skin fold thickness (mm)
5. 2-Hour serum insulin (mu U/ml)
6. Body mass index (BMI)
7. Diabetes pedigree function
8. Age (years)

**Download the dataset**: [pima-indians-diabetes.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)

Place the file in the same directory as your Python script.

## 🚀 Usage

### Basic Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/keras-diabetes-prediction.git
cd keras-diabetes-prediction
```

2. Ensure the dataset file `pima-indians-diabetes.csv` is in the project directory

3. Run the script:
```bash
python diabetes_prediction.py
```

### Code Example

```python
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Load dataset
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
x = dataset[:,0:8]  # Features
y = dataset[:,8]    # Labels

# Create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x, y, epochs=15, batch_size=10, verbose=2)

# Evaluate Model
score = model.evaluate(x, y)
print(f"Accuracy: {score[1]*100:.2f}%")

# Make predictions
predictions = model.predict(x)
```

## Configuration Options

### Model Architecture

You can modify the neural network structure:

```python
model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))  # Increase neurons
model.add(Dense(12, activation='relu'))               # Add more layers
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

### Training Parameters

Adjust these hyperparameters for different results:

```python
model.fit(
    x, y,
    epochs=20,        # Increase for more training
    batch_size=32,    # Larger batch size for faster training
    validation_split=0.2,  # Add validation data
    verbose=1         # Show progress bar
)
```

### Optimizer Options

Try different optimizers:

```python
model.compile(
    loss='binary_crossentropy',
    optimizer='sgd',      # Or 'rmsprop', 'adagrad'
    metrics=['accuracy']
)
```

## Expected Output

```
Epoch 1/15
77/77 - 0s - loss: 5.4219 - accuracy: 0.6445
Epoch 2/15
77/77 - 0s - loss: 1.1872 - accuracy: 0.6536
...
Epoch 15/15
77/77 - 0s - loss: 0.5234 - accuracy: 0.7461

24/24 [==============================] - 0s 1ms/step
accuracy 74.61
```

The model typically achieves around **70-78% accuracy** on this dataset.

## Troubleshooting

### Common Issues

**Problem**: `FileNotFoundError: pima-indians-diabetes.csv`
- **Solution**: Make sure the dataset file is in the same directory as your script

**Problem**: `ModuleNotFoundError: No module named 'tensorflow'`
- **Solution**: Install TensorFlow: `pip install tensorflow`

**Problem**: Low accuracy (below 60%)
- **Solution**: Try increasing epochs, adjusting learning rate, or normalizing the data

**Problem**: Model overfitting
- **Solution**: Add validation split and implement early stopping:
```python
from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=3)
model.fit(x, y, epochs=50, validation_split=0.2, callbacks=[early_stop])
```

### GPU Support

To use GPU acceleration (if available):
```python
# TensorFlow will automatically use GPU if available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

## Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

### Ideas for Contributions

- Add data preprocessing/normalization
- Implement cross-validation
- Add visualization of training history
- Create a prediction interface
- Improve model architecture
- Add more evaluation metrics (precision, recall, F1-score)

##  License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Additional Resources

- [Keras Documentation](https://keras.io/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Pima Indians Diabetes Dataset Info](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

## Author

Your Name - [GitHub Profile](https://github.com/Gerald-mut)

---

**Note**: This is a learning project. For medical applications, consult healthcare professionals and use properly validated models. 
