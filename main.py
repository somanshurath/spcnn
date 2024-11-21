Number_of_Training_Samples = 10000
Number_of_Testing_Samples = 1000
Matrix_Size = 15
Max_Sequence_Length = 10
Epochs = 50
Full_Length = 7
Max_Tries_Full_Sequence_Prediction = 5 * Full_Length

from generate_data import *
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Training
A, B, Y = generate_dataset(Number_of_Training_Samples, Matrix_Size, Max_Sequence_Length)
X = np.stack([A, B], axis=-1)

print("Training phase...")

model = Sequential([
    Input(shape=(Matrix_Size, Matrix_Size, 2)),
    Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
    GlobalAveragePooling2D(),
    Dense(64, activation='relu', kernel_initializer='he_normal'),
    Dense(32, activation='relu', kernel_initializer='he_normal'),
    Dense(16, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X, Y, epochs=Epochs, batch_size=32, validation_split=0.2, verbose=2)

# Testing
print("Testing phase for next-in-sequence operators...")
Correct_Predictions = 0
A, B, Y = generate_dataset(Number_of_Testing_Samples, Matrix_Size, Max_Sequence_Length, fixed_length=False)
for test in range(Number_of_Testing_Samples):
    a, b, y = A[test], B[test], Y[test]
    predicted_index = np.argmax(model.predict(np.stack([[a], [b]], axis=-1), verbose=0))
    if y == predicted_index:
        Correct_Predictions += 1
    print(f"Test {test + 1} complete. Prediction: {Functions[predicted_index]}, Actual: {Functions[y]}")
print("Correct next-in-sequence predictions in testing =",
      f"{Correct_Predictions / Number_of_Testing_Samples * 100 : 0.2f}%")

print("Testing phase for Full Sequence Prediction...")
Completed_Predictions = 0
Sum_Full_Lengths = 0
Report = ["Failed", "Successful"]
# fixed_length = True creates datasets of exactly Full_Length sequences, switching to False generates all sizes starting at 1.
A, B, Y = generate_dataset(Number_of_Testing_Samples, Matrix_Size, Full_Length, fixed_length=True)
for test in range(Number_of_Testing_Samples):
    flag = 0
    a, b, y = A[test], B[test], Y[test]
    predicted_index = np.argmax(model.predict(np.stack([[a], [b]], axis=-1), verbose=0))
    count = 0
    while count < Max_Tries_Full_Sequence_Prediction:
        function = Functions[predicted_index]
        operator, se = function.split()
        a = Operators[operator](a, SE[int(se) - 1])
        if np.array_equal(a, b):
            Completed_Predictions += 1
            flag = 1
            Sum_Full_Lengths += count + 1
            break
        count += 1
        predicted_index = np.argmax(model.predict(np.stack([[a], [b]], axis=-1), verbose=0))
    print(f"Test {test + 1} complete. Termination {Report[flag]}")
print(f"Full sequences predicted: {Completed_Predictions / Number_of_Testing_Samples * 100 : 0.2f}%")
print(
    f"Average ratio of fully predicted sequence lengths to their actual lengths: {(Sum_Full_Lengths / Completed_Predictions) / Full_Length}")
