import numpy as np

alpha = 0.1
weights = np.array([0.5, 0.48, -0.7])
Light = np.array([[1, 0, 1],
                  [0, 1, 1],
                  [0, 0, 1],
                  [1, 1, 1],
                  [0, 1, 1],
                  [1, 0, 1]])
WalkStop = np.array([0, 1, 0, 1, 1, 0])

for iteration in range(40):
    CommonError = 0
    for rowIndex in range(len(WalkStop)):
        input = Light[rowIndex]
        true = WalkStop[rowIndex]

        prediction = np.dot(input, weights)

        error = (prediction - true) ** 2
        CommonError += error

        delta = prediction - true

        weights = weights - input * delta * alpha
        print("Prediction: " + str(prediction))

    print("Error: " + str(CommonError) + "\n")
