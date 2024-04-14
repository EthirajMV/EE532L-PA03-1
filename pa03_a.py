# EE532L - Deep Learning for Healthcare - Programming Assignment 03
# Authors: Jibitesh Saha, Sasidhar Alavala, Subrahmanyam Gorthi
# Important: Please do not change/rename the existing function names and write your code only in the place where you are asked to do it.


########################################################## Can be modified ##############################################################
# You can import libraries as per your need
import pandas as pd
import sys
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# Write your code below so that it returns y_test_pred

    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def regress_fit(X_train, y_train, X_test, X_val, y_val, epochs=1000, learning_rate=0.01):

    #Normalisation
    for i in range(8):
        X_train[i] = (X_train[i] - np.min(X_train[i], axis=0)) / (np.max(X_train[i], axis=0) - np.min(X_train[i], axis=0))
        X_test[i] = (X_test[i] - np.min(X_test[i], axis=0)) / (np.max(X_test[i], axis=0) - np.min(X_test[i], axis=0))
        X_val[i] = (X_val[i] - np.min(X_val[i], axis=0)) / (np.max(X_val[i], axis=0) - np.min(X_val[i], axis=0))
    input_neurons = X_train.shape[0]

    hidden_neurons = 3
    output_neurons = 1

    # Weights and Biases
    np.random.seed(0)
    w1 = np.random.rand(hidden_neurons, input_neurons)
    w2 = np.random.rand(output_neurons, hidden_neurons)
    b1 = np.random.rand(hidden_neurons, 1)
    b2 = np.random.rand(output_neurons, 1)

    metrics_train = []
    metrics_val = []

    for epoch in range(epochs):
        # Feed Forward
        v1 = np.dot(w1, X_train) + b1
        y1 = sigmoid(v1)

        v2 = np.dot(w2, y1) + b2
        output = sigmoid(v2)

        # Backpropagation
        error = y_train.reshape(1, -1) - output
        d_output = error * ((output)*(1-(output)))
        error_hidden_layer = np.dot(w2.T, d_output)
        d_hidden_layer = error_hidden_layer * ((y1)*(1-(y1)))


        # Update weights and biases
        w2 += np.dot(d_output, y1.T) * learning_rate
        w1 += np.dot(d_hidden_layer, X_train.T) * learning_rate
        b2 += np.sum(d_output, axis=1, keepdims=True) * learning_rate
        b1 += np.sum(d_hidden_layer, axis=1, keepdims=True) * learning_rate

        # Loss
        train_loss = np.mean(np.abs(error))
        train_pred = (output >= 0.5).astype(int)
        train_accuracy = accuracy_score(y_train, train_pred.flatten())
        train_precision = precision_score(y_train, train_pred.flatten(), zero_division=1)
        train_recall = recall_score(y_train, train_pred.flatten())
        tn, fp, fn, tp = confusion_matrix(y_train, train_pred.flatten()).ravel()
        train_specificity = tn / (tn + fp)
        train_f1 = f1_score(y_train, train_pred.flatten())
        train_iou = tp / (tp + fp + fn)

        v1_val = np.dot(w1, X_val) + b1
        y1_val = sigmoid(v1_val)
        v2_val = np.dot(w2, y1_val) + b2
        output_val = sigmoid(v2_val)
        val_loss = np.mean(np.abs(y_val.reshape(1, -1) - output_val))
        val_pred = (output_val >= 0.5).astype(int)
        val_accuracy = accuracy_score(y_val, val_pred.flatten())
        val_precision = precision_score(y_val, val_pred.flatten(), zero_division=1)
        val_recall = recall_score(y_val, val_pred.flatten())
        tn, fp, fn, tp = confusion_matrix(y_val, val_pred.flatten()).ravel()
        val_specificity = tn / (tn + fp)
        val_f1 = f1_score(y_val, val_pred.flatten())
        val_iou = tp / (tp + fp + fn)

        metrics_train.append({'loss': train_loss,
                              'accuracy': train_accuracy,
                              'precision': train_precision,
                              'recall': train_recall,
                              'specificity': train_specificity,
                              'f1_score': train_f1,
                              'iou': train_iou})

        metrics_val.append({'loss': val_loss,
                            'accuracy': val_accuracy,
                            'precision': val_precision,
                            'recall': val_recall,
                            'specificity': val_specificity,
                            'f1_score': val_f1,
                            'iou': val_iou})

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

    metrics_names = ['loss', 'accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'iou']

    for i, metric_name in enumerate(metrics_names):
        plt.plot([metric[metric_name] for metric in metrics_train], label='Training')
        plt.plot([metric[metric_name] for metric in metrics_val], label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel(metric_name.capitalize())
        plt.legend()
        plt.show()

    # Predict on test data
    v1_test = np.dot(w1, X_test) + b1
    y1_test = sigmoid(v1_test)
    v2_test = np.dot(w2, y1_test) + b2
    y_test_pred = sigmoid(v2_test)
    y_test_pred = np.where(y_test_pred >= 0.5, 1, 0)
    return y_test_pred.T
    
###########################################################################################################################################


########################################################## Cannot be modified ##############################################################
# Load the dataset
def load_and_fit():
    df = pd.read_csv("data/diabetes.csv")
    df = df.sample(frac=1).reset_index(drop=True)
    # print(df.shape)
    X = df.drop("Outcome", axis=1)
    X2 = np.array(X)
    X2 = X2.T

    y = df["Outcome"]
    X_train = X2[:,:614]
    X_val = X2[:,614:691]
    X_test = X2[:,691:]
    y_train = y[:614].values
    y_val = y[614:691].values
    y_test = y[691:].values

    # Fit the model
    y_test_pred = regress_fit(X_train, y_train, X_test, X_val, y_val)

    # Evaluate the accuracy
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy:.5f}")
    return round(test_accuracy, 5)

a = load_and_fit()
