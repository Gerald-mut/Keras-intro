# Learning Keras - Diabetes Prediction Project

A beginner's journey into neural networks using Keras/TensorFlow. 
## About This Project

This is my learning project as I explore the fundamentals of deep learning with Keras. The goal was to understand how to build, compile, train, and evaluate a simple neural network from scratch. I chose the diabetes prediction task because it's a clear binary classification problem with real-world medical data.

## What I Learned

* **Building Neural Networks** - How to create a Sequential model with multiple layers
* **Model Compilation** - Understanding optimizers, loss functions, and metrics
* **Training Process** - How the fit() method trains the model with epochs and batches
* **Model Evaluation** - Measuring accuracy and making predictions
* **Working with Real Data** - Loading and preparing medical datasets

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

## Usage

### Running the Project

1. Clone the repository:
```bash
git clone https://github.com/Gerald-mut/learning-keras.git
cd learning-keras
```

2. Ensure the dataset file `pima-indians-diabetes.csv` is in the project directory

3. Run the script:
```bash
python diabetes_prediction.py
```

### My Code
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

## Understanding the Code

### Step 1: Creating the Model
I built a 3-layer Sequential neural network:
- **Input layer**: 12 neurons, takes 8 features
- **Hidden layer**: 8 neurons
- **Output layer**: 1 neuron with sigmoid activation (for binary output)

### Step 2: Compiling the Model
I configured the learning process:
- **Optimizer**: Adam (adaptive learning rate)
- **Loss function**: Binary crossentropy (for yes/no classification)
- **Metrics**: Accuracy (to track performance)

### Step 3: Training the Model
The model learned from the data over 15 epochs with batches of 10 samples at a time.

### Step 4: Evaluation
I tested the model's accuracy on the same dataset and generated predictions.

## My Results
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

I achieved around **74% accuracy** on my first try!

## Things I Want to Try Next

- [ ] Add data normalization/standardization
- [ ] Split data into training and testing sets
- [ ] Implement cross-validation
- [ ] Try different architectures (more/fewer layers)
- [ ] Visualize training history with matplotlib
- [ ] Add dropout layers to prevent overfitting
- [ ] Experiment with different optimizers
- [ ] Calculate precision, recall, and F1-score

## Challenges I Faced

**Understanding activation functions**: It took me time to understand why sigmoid is used for the output layer in binary classification.

**Choosing hyperparameters**: I'm still learning how to pick the right number of epochs, batch size, and neurons per layer.

**Overfitting concerns**: I'm training and evaluating on the same data, which isn't ideal. Need to learn about train/test splits.

## Resources That Helped Me

- [Keras Documentation](https://keras.io/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Pima Indians Diabetes Dataset Info](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

## Acknowledgments

This project was inspired by various Keras tutorials for beginners. The Pima Indians Diabetes dataset is a classic dataset in machine learning education.

## Author

Gerald - [GitHub Profile](https://github.com/Gerald-mut)

Learning deep learning one model at a time!

---

**Note**: This is a learning project and my first neural network. The model is not intended for medical use and is purely for educational purposes.
