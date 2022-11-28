import os
import cv2 as cv
import tkinter  as tk
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from src import files, cnn, operations, knc

reloaded, reloaded_binary = cnn.load_models()

main_window = tk.Tk()
main_window.title("DOUIA")
main_window.geometry("550x500")

results = tk.StringVar()

def train_binary():
    operations.train_model(reloaded_binary, end_layer_=2)
    reloaded_binary.load_weights("./ClsKLData/kneeKL224/model/2_weight.hdf5")

def train_five_class():
    operations.train_model(reloaded_binary, end_layer=5)
    reloaded.load_weights("./ClsKLData/kneeKL224/model/5_weight.hdf5")

def predict_binary():
    operations.predict_binary(reloaded_binary)
    
    evalu = ''
    for eval in cnn.evaluation:
        evalu+=f'{eval}\n'

    results.set(evalu)

def predict_five_class():
    operations.predict_five_class(reloaded)
    
    evalu = ''
    for eval in cnn.evaluation:
        evalu+=f'{eval}\n'

    results.set(evalu)

def knn_binary():
    operations.knn_binary()

    evalu = ''
    for eval in knc.evaluation:
        evalu+=f'{eval}\n'

    results.set(evalu)

def knn_five_class():
    operations.knn_five_class()

    evalu = ''
    for eval in knc.evaluation:
        evalu+=f'{eval}\n'

    results.set(evalu)

#Definição dos elementos da interface
tk.Button(main_window, text="Create Dataset", command=lambda:operations.create_data_set()).pack()
tk.Button(main_window, text="Create Validation Dataset", command=lambda:operations.create_data_set_val('./ClsKLData/kneeKL224/val', './ClsKLData/kneeKL224')).pack()
tk.Button(main_window, text="Create new Model Binario", command=lambda:operations.train_validate(amount_=5, end_layer_=2)).pack()
tk.Button(main_window, text="Create new Model 5-Classes", command=lambda:operations.train_validate(amount_=5, end_layer_=5)).pack()
tk.Button(main_window, text="Train Model Binary", command=lambda:train_binary()).pack()
tk.Button(main_window, text="Train Model 5-Classes", command=lambda:train_five_class()).pack()
tk.Button(main_window, text="Predict Binary and Show Resultados", command=lambda: predict_binary()).pack()
tk.Button(main_window, text="Predict 5-class and Show Resultados", command=lambda: predict_five_class()).pack()
tk.Button(main_window, text="KNN Binary and Show Resultados", command=lambda: knn_binary()).pack()
tk.Button(main_window, text="KNN 5-class and Show Resultados", command=lambda: knn_five_class()).pack()
tk.Label(main_window, textvariable=results, fg='blue').pack()
main_window.mainloop()