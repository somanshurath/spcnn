from structuring_elements import *
from scipy.ndimage import binary_dilation, binary_erosion

Number_of_Training_Samples = 10000
Matrix_Size = 10
Max_Sequence_Length = 5

Functions = [f"Dilation {i}" for i in range(1, 9)] + [f"Erosion {i}" for i in range(1, 9)]
Operators = {"Dilation": binary_dilation, "Erosion": binary_erosion}


def generate_dataset(samples, size, max_sequence_length, fixed_length=False):
    a, b, Y = [], [], []
    for sample in range(samples):
        A = np.random.randint(2, size=(size, size))
        # Preventing a zero matrix, unlikely after 2 random generations
        if not A.any(): A = np.random.randint(2, size=(size, size))
        for iteration in range(max_sequence_length):
            index = np.random.randint(16)
            function = Functions[index]
            if iteration == 0 or fixed_length == False:
                a.append(A)
                Y.append(index)
            operator, se = function.split()
            B = Operators[operator](A, SE[int(se) - 1])
            if not B.any():
                B = Operators["Dilation"](A, SE[int(se) - 1])
            A = B
        if fixed_length == False:
            for iteration in range(max_sequence_length - 1): b.append(B)
        b.append(B)
    Y = np.array(Y)
    return a, b, Y
