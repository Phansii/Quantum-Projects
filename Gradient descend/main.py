import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from qiskit import QuantumCircuit
import qiskit as qc
from qiskit.quantum_info import Statevector

data = pd.read_csv("data.txt")

features = data[["Light","CO2"]]
labels = data["Occupancy"]

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2, random_state=47, stratify=labels)

# min max scaling
maximum = train_features.max(axis=0)
minimum = train_features.min(axis=0)

# min max scaling function
def minmax(feature):
    scaled_feature = np.pi * ((feature-minimum)/(maximum-minimum))
    return scaled_feature

train_features = minmax(train_features)
test_features = minmax(test_features)

def model(point, trainable_angles):
    qc = QuantumCircuit(2)
    qc.ry(point["Light"], 0)
    qc.rx(point["CO2"], 1)

    qc.ry(trainable_angles[0], 0)
    qc.ry(trainable_angles[1], 1)
    qc.cx(0, 1)

    state = Statevector.from_instruction(qc)
    probabilities = state.probabilities([1])
    return probabilities

def cost_function(train_features, angle, train_labels):
    cost = 0
    for i in range(len(train_features)):
        prediction = model(train_features.iloc[i], angle)[1]
        actual = train_labels.iloc[i]
        cost = cost + (prediction - actual) ** 2

    cost = cost / len(train_features)
    return cost

def gradient(train_features, angle, train_labels):
    gradient = 0
    diff = 0.001
    angle_diff_0 = angle.copy()
    angle_diff_0[0] += diff

    angle_diff_1 = angle.copy()
    angle_diff_1[1] += diff
    
    gradient_0 = (cost_function(train_features, angle_diff_0, train_labels) - cost_function(train_features, angle, train_labels))/(diff)
    gradient_1 = (cost_function(train_features, angle_diff_1, train_labels) - cost_function(train_features, angle, train_labels))/(diff)

    return [gradient_0, gradient_1]

def gradient_descend(train_features, angle, train_labels):
    learning_rate = 0.1

    new_angle = np.array(angle) - learning_rate * np.array(gradient(train_features, angle, train_labels))

    return new_angle


train_angle = [0.2,0.8]
for i in range(10):
    train_angle = gradient_descend(train_features, train_angle, train_labels)

print(model(test_features.iloc[0], train_angle))
print(test_labels.iloc[0])