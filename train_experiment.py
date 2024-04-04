# version 1.1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neural_net import NeuralNetwork
from operations import *

def load_dataset(csv_path, target_feature):
    dataset = pd.read_csv(csv_path)
    t = np.expand_dims(dataset[target_feature].to_numpy().astype(float), axis=1)
    X = dataset.drop([target_feature], axis=1).to_numpy()
    return X, t

X, y = load_dataset("Q2\data\wine_quality.csv", "quality")

n_features = X.shape[1]
epochs = 500
epoch_values=[]

test_split = 0.2

for i in range(5):
    net = NeuralNetwork(n_features, [64,32,1], [ReLU(), ReLU(), Identity()], MeanSquaredError(), learning_rate=0.001)

    x_idx_left,x_idx_right=int(0.2*i*X.shape[0]),int(0.2*(i+1)*X.shape[0])
    y_idx_left,y_idx_right=int(0.2*i*y.shape[0]),int(0.2*(i+1)*y.shape[0])

    X_train=np.vstack((X[:x_idx_left],X[x_idx_right:]))
    X_test=X[x_idx_left:x_idx_right]
    y_train=np.vstack((y[:y_idx_left],y[y_idx_right:]))
    y_test=y[y_idx_left:y_idx_right]
# X_train = X[:int((1 - test_split) * X.shape[0])]
# X_test = X[int((1 - test_split) * X.shape[0]):]
# y_train = y[:int((1 - test_split) * y.shape[0])]
# y_test = y[int((1 - test_split) * y.shape[0]):]

    trained_W, epoch_losses = net.train(X_train, y_train, epochs)
    epoch_values.append(epoch_losses)
    print("Error on test set: {}".format(net.evaluate(X_test, y_test, mean_absolute_error)))

average_loss=[]
for i in range(epochs):
    current_loss=0
    for j in range(5):
        current_loss+=epoch_values[j][i]
    average_loss.append(current_loss/5)


final_error_arr=[]
for i in range(5):
    final_error_arr.append(epoch_values[i][-1])
final_mean_loss=np.mean(final_error_arr)
final_sd=np.std(final_error_arr)


print("\nAverage error over all folds: ", final_mean_loss)
print("Standard deviation of errors: ", final_sd)
plt.plot(np.arange(0, epochs), average_loss)
plt.xlabel("Epoch")
plt.ylabel("Average training loss")
plt.show()