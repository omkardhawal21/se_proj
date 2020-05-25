from tkinter import *
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style



def Predict_result():
    data = pd.read_csv("student-mat.csv", sep=";")

    data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

    predict = "G3"
    X = np.array(data.drop([predict], 1))
    Y = np.array(data[predict])

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    with open("studentmodel.pickle", "wb") as f:
        pickle.dump(linear, f)

    pickle_in = open("studentmodel.pickle", "rb")
    linear = pickle.load(pickle_in)

    print('Coefficient: \n', linear.coef_)
    print('Intercept: \n', linear.intercept_)
    m = int(t1.get())
    n = int(t2.get())
    o = int(t3.get())
    p = int(t4.get())
    r = int(t5.get())
    e = [[m, n, o, p, r]]
    predictions = linear.predict(e)
    # for x in range(len(predictions)):
    print(predictions, e)
    q = acc
    t6.insert(END, q*100)
    u = predictions*4
    t7.insert(END, u)

def clear():
    t1.delete(0, END)
    t2.delete(0, END)
    t3.delete(0, END)
    t4.delete(0, END)
    t5.delete(0, END)
    t6.delete(0, END)
    t7.delete(0, END)

root = Tk()
root.title("Student Performance Prediction")
G1 = IntVar()
G2 = IntVar()
G3 = IntVar()
studytime = IntVar()
failures = IntVar()
absences = IntVar()
accuracy = IntVar()
result = IntVar()
m = IntVar()
n = IntVar()
o = IntVar()
p = IntVar()
r = IntVar()
u = IntVar()

l1 = Label(root, text="Enter Unit1 marks(out of 20)",bg="orange")
l1.grid(row=0, column=0)
l2 = Label(root, text="Enter Unit2 marks(out of 20)",bg ="orange")
l2.grid(row=1, column=0)
l3 = Label(root, text="Enter Studytime(in hrs)",bg ="orange")
l3.grid(row=2, column=0)
l4 = Label(root, text="Enter Failures",bg ="orange")
l4.grid(row=3, column=0)
l5 = Label(root, text="Enter Absences",bg ="orange")
l5.grid(row=4, column=0)
l6 = Label(root, text="--------------------------")
l6.grid(row=5, column=1)
l7 = Label(root, text="Semester Result(Out of 80)",bg ="yellow")
l7.grid(row=10, column=0)
l8 = Label(root, text="Accuracy(in %)",bg ="yellow")
l8.grid(row=7, column=0)

b1 = Button(root, text="Predict_result", command=Predict_result,bg="cyan")
b1.grid(row=6, column=1)
b2 = Button(root, text="Clear", command=clear)
b2.grid(row=10, column=3)

t1 = Entry(root)
t1.grid(row=0, column=2)
t2 = Entry(root)
t2.grid(row=1, column=2)
t3 = Entry(root)
t3.grid(row=2, column=2)
t4 = Entry(root)
t4.grid(row=3, column=2)
t5 = Entry(root)
t5.grid(row=4, column=2)
t7 = Entry(root)
t7.grid(row=10, column=2)
t6 = Entry(root)
t6.grid(row=7, column=2)
root.mainloop()
