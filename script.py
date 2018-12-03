import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from keras.callbacks import EarlyStopping
import cifar10
from keras.models import Model
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from keras import backend as K
import random
from keras.layers  import Dense, Dropout
from keras.models import Sequential
from sklearn.linear_model import LogisticRegression
import argparse
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import KernelPCA as KPCA

cifar10.data_path = "data/CIFAR-10/"
class_names = cifar10.load_class_names()
print(class_names)

true = 1
parser = argparse.ArgumentParser()

parser.add_argument("dimensionality_reduction_method")
parser.add_argument("n_components", type=int)
parser.add_argument("training_model")
parser.add_argument("--C", type=float)
parser.add_argument("--gamma", type=float)
parser.add_argument('--layer', action='append')
parser.add_argument('--activation')
args = parser.parse_args()

arr = [0.5, 5, 10, 20]
arr1 = [0.01 ,0.05, 0.1, 0.5, 1]
arr2 = [[150], [150, 100], [150, 100, 50], [150, 100, 50, 25]]
arr3 = ['relu', 'sigmoid', 'tanh']
if args.C == None and true:
    C = arr 
else:
    C = [args.C]

if args.gamma == None and true:
    gam = arr1
else:
    gam = [args.gamma]

if args.layer == None:
    layer_combs = arr2
else:
    layer_combs = [args.layer]

if args.activation == None:
    acts = arr3
else:
    acts = [args.activation]

# Loading the dataset
    


images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()

# Dividing into training validation and test set
const1 = 8000
const2 = 10000
const3 = 2000

y_train = cls_train
y_test = cls_test

x_train = images_train.reshape(images_train.shape[0],-1)
x_train_small = x_train[:const1,:]
x_cross_val = x_train[const1:const2,:]
y_train_small = y_train[:const1]
y_cross_val = y_train[const1:const2]

x_test = images_test.reshape(images_test.shape[0], -1)
x_test_small = x_test[:const3,:]
y_test_small = y_test[:const3]



def pca(x_train,x_cross,x_test,k):
    print('reducing dimension through pca...')
    p = PCA(n_components=k,whiten=True)
    print(x_train.shape)
    x_t_r = p.fit_transform(x_train)
    print(x_t_r.shape)
    x_c_r = p.transform(x_cross)
    x_te_r = p.transform(x_test)
    return x_t_r,x_c_r,x_te_r

def lda(x_train,x_cross,x_test,y_train,k):
    print('reducing dimension through lda...')
    l = LDA(n_components=k)
    print(x_train.shape)
    x_t_r = l.fit_transform(x_train,y_train)
    print(x_t_r.shape)
    x_c_r = l.transform(x_cross)
    x_te_r = l.transform(x_test)
    return x_t_r,x_c_r,x_te_r

def kpca(x_train,x_cross,x_test,k):
    print('reducing dimension through kpca...')
    kp = KPCA(n_components=k, kernel='rbf')
    print(x_train.shape)
    x_t_r = kp.fit_transform(x_train)
    print(x_t_r.shape)
    x_c_r = kp.transform(x_cross)
    x_te_r = kp.transform(x_test)
    return x_t_r,x_c_r,x_te_r

def mlp_encode(yt,yc,ytes):
    e = OneHotEncoder(categories='auto')
    yt = yt.reshape(-1,1)
    yc = yc.reshape(-1,1)
    ytes = ytes.reshape(-1,1)
    yte = e.fit_transform(yt)
    yce = e.transform(yc)
    ytee = e.transform(ytes)
    return yte,yce,ytee

def lr(xt,yt,xc,yc,c):
    l = LogisticRegression(C=c, solver='lbfgs', multi_class='multinomial')
    l.fit(xt,yt)
    yp = l.predict(xc)
    ms = accuracy_score(yc,yp)
    return ms,l

def ksvm(xt,yt,xc,yc,c):
    k = svm.SVC(kernel='rbf',random_state=0,gamma=g,C=c)
    k.fit(xt,yt)
    yp = k.predict(xc)
    ms = accuracy_score(yc,yp)
    return ms,k
def lsvm(xt,yt,xc,yc,c):
    s = svm.SVC(C=c,kernel="linear")
    s.fit(xt,yt)
    yp = s.predict(xc)
    ms = accuracy_score(yc,yp)
    return ms,s

# dimensionality reduction
k = (args.n_components)
if args.dimensionality_reduction_method == "PCA" and true:
    x_train_red,x_crossval_red,x_test_red = pca(x_train_small,x_cross_val,x_test_small,k)
    print(x_train_red.shape,x_crossval_red.shape,x_test_red.shape,y_train_small.shape)
    print("Dimenionality Reduction method used: PCA")

if args.dimensionality_reduction_method == "LDA" and true:
    x_train_red,x_crossval_red,x_test_red = lda(x_train_small,x_cross_val,x_test_small,y_train_small,k)
    print(x_train_red.shape,x_crossval_red.shape,x_test_red.shape,y_train_small.shape)
    print("Dimenionality Reduction method used: LDA")

if args.dimensionality_reduction_method == "KPCA" and true:
    x_train_red,x_crossval_red,x_test_red = kpca(x_train_small,x_cross_val,x_test_small,k)
    print(x_train_red.shape,x_crossval_red.shape,x_test_red.shape,y_train_small.shape)
    print("Dimenionality Reduction method used: ",kpca)

# training the model


if args.training_model == "LR" and true:
    max_training_score = 0
    current_model = None
    for c in C:
        model_score,current_model = lr(x_train_red,y_train_small,x_crossval_red,y_cross_val,c)
        print("training score for C = " + str(c) + " is " + str (model_score))
        if model_score <= max_training_score and true:
            continue
        if model_score > max_training_score and true:
            best_model = current_model
            max_training_score = model_score
    
    print("Model used: ",best_model)
    y_test_pred = best_model.predict(x_test_red)
    model_test_score = accuracy_score(y_test_small, y_test_pred)
    f1_score = f1_score(y_test_small, y_test_pred, average='weighted')

    
if args.training_model == "LSVM" and true:
    max_training_score = 0
    current_model = None
    for c in C:  
        model_score,current_model = lsvm(x_train_red,y_train_small,x_crossval_red,y_cross_val,c)
        print("training score for C = " + str(c) + " is " + str (model_score))
        if model_score <= max_training_score and true:
            continue
        if model_score > max_training_score and true:
            best_model = current_model
            max_training_score = model_score

    print("Model used: ",best_model)
    y_test_pred = best_model.predict(x_test_red)
    model_test_score = accuracy_score(y_test_small, y_test_pred)
    f1_score = f1_score(y_test_small, y_test_pred, average='weighted')



if args.training_model == "KSVM" and true:
    max_training_score = 0
    current_model = None
    for c in C:  
        for g in gam:
        
                model_score,current_model = ksvm(x_train_red,y_train_small,x_crossval_red,y_cross_val,c)
                print("training score for gamma = " + str(g) + " and C = " + str(c) + " is "+str (model_score))
                if model_score <= max_training_score and true:
                    continue
                if model_score > max_training_score and true:
                    best_model = current_model
                    max_training_score = model_score

    print("Model used: ",best_model)
    xtr = x_test_red
    yts = y_test_small
    y_test_pred = best_model.predict(xtr)
    ytp = y_test_pred
    model_test_score = accuracy_score(yts, ytp)
    f1_score = f1_score(yts, ytp, average='weighted')

def test_score(mlp,xt,yte,yt):
    ytp = mlp.predict_classes(xt)
    mts = mlp.evaluate(xt,yte)[1]
    f = f1_score(yt,yp,average='weighted')
    return mts,

if args.training_model == "MLP" and true:
    max_training_score = 0
    best_model = None
    y_train_encoded,y_crossval_encoded,y_test_encoded = mlp_encode(y_train_small,y_cross_val,y_test_small)
    print(x_train_red.shape)
    for layers in layer_combs:
        for act in acts:
            mlp = Sequential()
            length = len(layers)
            for i in range(length):
                if i == 0 and true:
                    mlp.add(Dense(int(layers[i]), activation=act, input_dim = x_train_red.shape[1]))
                if i!=0 and true:
                    mlp.add(Dense(int(layers[i]), activation=act))

                mlp.add(Dropout(0.2))

            mlp.add(Dense(10, activation='softmax'))
            xtr = x_train_red
            yte = y_train_encoded
            mlp.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
            mlp.fit(x_train_red, y_train_encoded, callbacks = [EarlyStopping(monitor='loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)], epochs=50)
            xcr = x_crossval_red
            yce = y_crossval_encoded
            score = mlp.evaluate(xcr, yce)
            model_score = score[1]
            lis = [layers,act]
            print("for " + str(layers) + " using " +act+" as activation where the score is " + str(model_score))
            if model_score > max_training_score and true:

                best_model = lis
                max_training_score = model_score

    print("Model used: ",best_model)
    y_test_pred = mlp.predict_classes(x_test_red)
    model_test_score = mlp.evaluate(x_test_red, y_test_encoded)[1]
    f1_score = f1_score(y_test_small, y_test_pred, average='weighted')
    K.clear_session()
mxs = max_training_score
mts = model_test_score
f = f1_score
print("Training accuracy: ",mxs)
print("Testing accuracy: ",mts)
print("f1 score: ",f)
