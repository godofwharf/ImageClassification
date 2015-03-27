import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import time
import random
import csv
import pylab as pl
import pickle
from os import listdir
from os.path import isfile, join
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import svm
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import chi2_kernel
from scipy.cluster.vq import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA, pca


# Setting paths to training and test data
PATH = "C:\\Users\\User\\Desktop\\TagMe!-Data"
TRAIN_PATH = PATH + "\\" + "Train"
TEST_PATH = PATH + "\\" + "Validation"
TRAIN_IMAGES = TRAIN_PATH + "\\" + "Images"
TRAIN_LABELS = TRAIN_PATH + "\\" + "Labels.txt"
TRAIN_CSV = TRAIN_PATH + "\\" + "Train.csv"
TEST_IMAGES = TEST_PATH + "\\" + "Images"
TEST_CSV = TEST_PATH + "\\" + "Test.csv"

# Setting global variables
model = None
labels = {}
k_means = None

# No. of clusters
k = 32

# Dimension of SURF
dim = 128

# Function to write k_means object to file
def saveToFile(obj):
    with open(join(PATH, 'k_means.obj'), 'wb') as fp:
        pickle.dump(obj, fp)
        
# Function to load k_mean object from file
def loadFromFile():
    with open(join(PATH, 'k_means.obj'), 'rb') as fp:
        return pickle.load(fp)

# Class for SURF descriptor
class Descriptor:

    def __init__(self, x, y, descriptor):
        # setting x-coordinate
        self.x = x
        # setting y-coordinate
        self.y = y
        # normalize SURF descriptor (L2 norm)
        self.descriptor = self.normalize(descriptor)

    # Function to perform L2 norm
    def normalize(self, descriptor):
        descriptor = np.array(descriptor)
        norm = np.linalg.norm(descriptor)

        # To avoid dividing by very small values
        if norm > 1.0:
            descriptor /= float(norm)

        return descriptor


# Class for description of an image
class imageDescriptors:

    '''
    Input: 
    descriptors - a set of SURF descriptors
    label - label for the image
    width - width of the image
    height - height of the image
    '''
    def __init__(self, descriptors, label, width, height):
        self.descriptors = descriptors
        self.label = label
        self.width = width
        self.height = height
        

# Function to read labels for training images
# labels are read for each image and a dictionary is created as pairs of <img_label, class_label>
def readLabels(TRAIN_LABELS):
    
    with open(TRAIN_LABELS, 'rU') as fp:
        for line in fp:
             fname, label = line.split(" ")
             labels[fname] = label.strip(' \t\n\r')

# Function to compute SURF descriptors for a given image
def doSURF(fileName):
    # read image
    img = cv2.imread(fileName)
    
    # convert image to gray scale
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    '''
    creates surf detector
    hessianThreshold - determines no. of detected keypoints (lower the threshold, more the number of keypoints)
    value set to 500 based on suggestions in paper
    extended - if set to True, computes SURF of 128 dimension; otherwise 64 dimension
    '''
    detector = cv2.SURF(hessianThreshold = 500, extended = True)
    
    # get keypoints and descriptors for them
    kp, des = detector.detectAndCompute(grey, None)
    
    # if there are no keypoints found, enhance the contrast of the image and try again
    if des == None:
        equ = cv2.equalizeHist(grey)
        kp, des = detector.detectAndCompute(equ, None)
    
    return kp, des

# Function to compute SIFT descriptors for an image
def doSIFT(fileName):
    img = cv2.imread(fileName)
    grey= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(grey, None)
    
    return kp, des

# build spatial pyramids of an image based on the level attribute
''' 
Input:
descriptorsOfImage - SURF descriptors for an image
level - no. of levels to construct pyramids on
'''
def buildVLADForEachImageAtDifferentLevels(descriptorsOfImage, level):

    # Set width and height
    width = descriptorsOfImage.width
    height = descriptorsOfImage.height
    # calculate width and height step
    widthStep = int(width / 2)
    heightStep = int(height / 2)

    descriptors = descriptorsOfImage.descriptors

    # level 1, a list with size = 4 to store histograms at different location
    VLADOfLevelOne = np.zeros((4, k, dim))
    for descriptor in descriptors:
        x = descriptor.x
        y = descriptor.y
        boundaryIndex = int(x / widthStep)  + int(y / heightStep)

        feature = descriptor.descriptor
        shape = feature.shape[0]
        feature = feature.reshape(1, shape)

        codes, distance = vq(feature, k_means.cluster_centers_)
        
        VLADOfLevelOne[boundaryIndex][codes[0]] += np.array(feature).reshape(shape) - k_means.cluster_centers_[codes[0]]
    
    
    for i in xrange(4):
        # Square root norm
        VLADOfLevelOne[i] = np.sign(VLADOfLevelOne[i]) * np.sqrt(np.abs(VLADOfLevelOne[i]))
        # Local L2 norm
        vector_norm = np.linalg.norm(VLADOfLevelOne[i], axis = 1)
        vector_norm[vector_norm < 1] = 1
        
        VLADOfLevelOne[i] /= vector_norm[:, None]
    
    # level 0
    VLADOfLevelZero = VLADOfLevelOne[0] + VLADOfLevelOne[1] + VLADOfLevelOne[2] + VLADOfLevelOne[3]
    # Square root norm
    VLADOfLevelZero = np.sign(VLADOfLevelZero) * np.sqrt(np.abs(VLADOfLevelZero))
    # Local L2 norm
    vector_norm = np.linalg.norm(VLADOfLevelZero, axis = 1)
    vector_norm[vector_norm < 1] = 1
    
    VLADOfLevelZero /= vector_norm[:, None]
    
    if level == 0:
        return VLADOfLevelZero

    elif level == 1:
        tempZero = VLADOfLevelZero.flatten() * 0.5
        tempOne = VLADOfLevelOne.flatten() * 0.5
        result = np.concatenate((tempZero, tempOne))
        # Global L2 norm
        norm = np.linalg.norm(result)
        if norm > 1.0:
            result /= norm
        return result
    else:
        return None

# Function to read images from training set and perform feature extraction

def prepareTrainingSet():
    # read labels for images and place in dictionary
    readLabels(TRAIN_LABELS)
    features = []
    sd = []
    allFiles = labels.keys()
    images = []
    global k_means
    for fn in allFiles:
        kp, des = doSURF(join(TRAIN_IMAGES, fn))
        descriptors = []
        for i in xrange(len(kp)):
            x,y = kp[i].pt
            descriptor = Descriptor(x, y, des[i])
            descriptors.append(descriptor)
        imgDescriptor = imageDescriptors(descriptors, fn, 240, 240)
        sd += des.tolist()
        images.append(imgDescriptor)
    print "Number of keypoints extracted from training set = %d" % len(sd)
    start = time.clock()
    k_means = MiniBatchKMeans(n_clusters = k, batch_size = 20000, n_init = 30)
    k_means.fit(sd)
    end = time.clock()
    print "Time for running 30 iterations of K-means for %d samples = %f" % (len(sd), end - start)
    
    # pickle k_means object
    saveToFile(k_means)
    
    for image in images:
        # get Vlad feature vector for each image
        vlad = buildVLADForEachImageAtDifferentLevels(image, 1)
        features.append((image.label, vlad))
    # write features to csv file
    with open(TRAIN_CSV, 'wb') as fp:
        rowwriter = csv.writer(fp, delimiter = ',')
        for a,b in features:
            grp = int(labels[a])
            assert grp >= 1 and grp <= 5
            rowwriter.writerow([a] + b.tolist() + [grp])

# Function to read test images and extract features
def prepareTestSet():
    features = []
    lbls = []
    X = []
    images = []
    allFiles = [f for f in listdir(TEST_IMAGES) if isfile(join(TEST_IMAGES,f))]
    
    for fn in allFiles:
        kp, des = doSURF(join(TEST_IMAGES, fn))
        
        descriptors = []
        for i in xrange(len(kp)):
            x,y = kp[i].pt
            descriptor = Descriptor(x, y, des[i])
            descriptors.append(descriptor)
            X.append(des[i])
            
        imgDescriptor = imageDescriptors(descriptors, fn, 240, 240)
        images.append(imgDescriptor)
        
    print "Number of extracted keypoints from test set = %d" % len(X)
    
    for image in images:
        vlad = buildVLADForEachImageAtDifferentLevels(image, 1)
        features.append(vlad.tolist())
        lbls.append(image.label)
        # since test set is huge, need to classify images in batches of 100
        if len(features) == 100:
            outputLabels(lbls, features)
            features = []
    
    '''
    with open(TEST_CSV, 'wb') as fp:
        rowwriter = csv.writer(fp, delimiter = ',')
        for a,b in features:
            rowwriter.writerow([a] + b.tolist())
    '''
# classifies images in batches and appends predictions to output file
def outputLabels(lbls, X):
    Y = np.array(X)
    Y = Y.astype(float)
    scale(Y, with_mean = True, with_std = True)
    preds = model.predict(Y)
    writePredictions(lbls, preds)

# Function to read csv file
def readCsv(fn, is_train):
    lbls = []
    X = []
    y = []
    with open(fn, 'rb') as fp:
        reader = csv.reader(fp, delimiter = ',')
        
        for row in reader:
            lbls.append(row[0])
            if is_train:
                features = [float(ele) for ele in row[1:-1]]
            else:
                features = [float(ele) for ele in row[1:]]
            X.append(features)
            if is_train:
                output_class = row[-1].strip(' \n\t\r')
                y.append(int(output_class))
    return lbls, X, y

# Function to perform parameter tuning
def exhaustiveGridSearch():
    
    # read training file
    lbls1, X, y = readCsv(TRAIN_CSV, True)
    
    X = np.array(X)
    X = X.astype(float)
    y = np.array(y)
    
    # scale features for zero mean and unit variance 
    scale(X, with_mean = True, with_std = True)
    
    # Split the dataset in two parts
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)
    # Set the parameters by cross-validation
    
    tuned_parameters = [ # {'kernel': ['rbf'], 'gamma': [2**pw for pw in xrange(-15,3)], 'C': [2**pw for pw in xrange(-5,16)]} ]
                    # {'kernel': ['linear'], 'C': [2**pw for pw in xrange(-5,16)]} ]
                    # {'kernel': ['poly'], 'C':[2**pw for pw in xrange(-5,16)], 'degree':[i for i in xrange(2,7)] }]
                      {'C':[2**pw for pw in xrange(-5,16)]}]
                    #  {'max_features': [val for val in xrange(10,110,10)], 'min_samples_split': [val for val in xrange(10,110,10)]} ]
                     
    scores = ['precision', 'recall']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        clf = GridSearchCV(svm.LinearSVC(C=1), tuned_parameters, cv=5, scoring=score)
        # clf = GridSearchCV(RandomForestClassifier(n_estimators = 200, random_state = 100), tuned_parameters, cv=5, scoring=score)
        clf.fit(X_train, y_train)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_estimator_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))
        print()
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

# Compute Histogram Intesection kernel
def hist_intersection(x, y):
    n_samples , n_features = x.shape
    K = np.zeros(shape=(n_samples, n_samples),dtype=np.float)
    
    for r in xrange(n_samples):
        for c in xrange(n_samples):
            K[r][c] = np.sum(np.minimum(x[r], y[c]))
    return K

# Function to perform cross validation on different models
def cross_validate(X, y):
    
    svc = svm.SVC(kernel='linear', C = 0.0625)
    
    lin_svc = svm.LinearSVC(C = 4.0, dual = False)
    
    rbf_svc = svm.SVC(kernel='rbf', gamma = 0.0009765625, C = 32.0)
    
    poly_svc = svm.SVC(kernel='poly', degree = 2 , C = 2048.0)
    
    hist_svc = svm.SVC(kernel = 'precomputed')
    chi2_svc = svm.SVC(kernel = 'precomputed')
    
    # random_forest = RandomForestClassifier(n_estimators = 200, max_features = 50, min_samples_split = 20, random_state = 100)
    # 5-fold cross validation
    for model in [svc, lin_svc, rbf_svc, poly_svc]:
        print model
        scores = cross_val_score(model, X, y, cv=10)
        print scores
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    print hist_svc
    K = hist_intersection(X, X)
    scores = cross_val_score(model, K, y, cv=10)
    print scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    print chi2_svc
    K = chi2_kernel(X, gamma = 0.3)
    scores = cross_val_score(model, K, y, cv=10)
    print scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    
# Function to write predictions to output file
# Notice that append mode is used
def writePredictions(labels, preds):
    TEST_OUTPUT = join(TEST_PATH, "preds.txt")
    
    with open(TEST_OUTPUT, 'a') as fp:
         writer = csv.writer(fp, delimiter = ' ')
         for label, pred in zip(labels, preds):
             writer.writerow([label, pred])
             

# Function to perform training and classification
def classify():
    # read training data
    lbls1, X, y = readCsv(TRAIN_CSV, True)
    # read test data
    lbls2, Y, z = readTestCsv(TEST_CSV)
    
    # Conversion to numpy arrays
    X = np.array(X)
    X = X.astype(float)
    y = np.array(y)
    
    Y = np.array(Y)
    Y = Y.astype(float)
    
    # perform feature scaling for zero mean and unit variance
    scale(X, with_mean = True, with_std = True)
    scale(Y, with_mean = True, with_std = True)
    
    lin_svc = svm.LinearSVC(C = 4.0, dual = False)
    lin_svc.fit(X, y)
    
    bestmodel = lin_svc
    preds = bestmodel.predict(Y)
    
    writePredictions(lbls2, preds)
    
    
# Function to plot confusion matrix on training data
def plotConfusionMatrix():
    # read training data
    lbls1, X, y = readCsv(TRAIN_CSV, True)
    
    # Split the data randomly into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100)

    # Run classifier
    classifier = svm.LinearSVC(C = 4.0, dual = False)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    
    labels = ['building', 'car', 'person', 'flower', 'shoe']
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    pl.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    pl.xlabel('Predicted')
    pl.ylabel('True')
    pl.show()
    
if __name__ == "__main__":
    prepareTrainingSet()
    
    k_means = loadFromFile()
    lbls1, X, y = readCsv(TRAIN_CSV, True)
    X = np.array(X)
    X = X.astype(float)
    y = np.array(y)
    scale(X, with_mean = True, with_std = True)
    lin_svc = svm.LinearSVC(C = 4.0, dual = False)
    lin_svc.fit(X, y)
    model = lin_svc
    prepareTestSet()
    
    # plotConfusionMatrix()
    # exhaustiveGridSearch()
    # classify()