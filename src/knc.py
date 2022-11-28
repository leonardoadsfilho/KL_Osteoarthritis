from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from src import files, cnn, operations
from datetime import datetime

evaluation = []

def train_validate(amount_=10000, binary_=False):

    data_set, data_set_label = files.read_data_set('./ClsKLData/kneeKL224/data_set/', amount=amount_, binary=binary_)
    data_set_raw = np.array(list(map(lambda data: cnn.extract_color_histogram(data), data_set)))
    data_set_feature = np.array(list(map(lambda data: cnn.image_to_feature_vector(data), data_set)))
    print(f'end: {datetime.now().strftime("%H:%M:%S")}')

    (trainRI, testRI, trainRL, testRL) = train_test_split(
        data_set_raw, data_set_label, test_size=0.25, random_state=42)
    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
        data_set_feature, data_set_label, test_size=0.25, random_state=42)

    evaluation.clear()

    evaluation.append(f'end: {datetime.now().strftime("%H:%M:%S")}')
    print('[Training]')
    print(f'start: {datetime.now().strftime("%H:%M:%S")}')
    print("[INFO] evaluating raw pixel accuracy...")
    model = KNeighborsClassifier()
    model.fit(trainRI, trainRL)
    acc = model.score(testRI, testRL)
    print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))
    evaluation.append("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))
    
    print("[INFO] evaluating histogram accuracy...")
    model = KNeighborsClassifier()
    model.fit(trainFeat, trainLabels)
    acc = model.score(testFeat, testLabels)
    print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))
    evaluation.append(f'end: {datetime.now().strftime("%H:%M:%S")}')
    print(f'end: {datetime.now().strftime("%H:%M:%S")}')

    evaluation.append("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))

    data_set_val, data_set_label_val = files.read_data_set_val('./ClsKLData/kneeKL224/data_set/', amount=amount_, binary=binary_)
    data_set_raw_val = np.array(list(map(lambda data: cnn.extract_color_histogram(data), data_set_val)))
    data_set_feature_val = np.array(list(map(lambda data: cnn.image_to_feature_vector(data), data_set_val)))

    (trainFeat_val, testFeat_val, trainLabels_val, testLabels_val) = train_test_split(
        data_set_feature_val, data_set_label_val, test_size=0.10, random_state=42)

    predicted = model.predict(trainFeat_val)

    print('[Validation]')    

    acc_knn = accuracy_score(trainLabels_val, predicted)

    print(acc_knn)

    evaluation.append(acc_knn)

    print(predicted)
    print(trainLabels_val)

    evaluation.append(predicted)
    evaluation.append(trainLabels_val)

    print('[Confusion Matrix]')

    c_matrix = confusion_matrix(trainLabels_val, predicted)
    
    print(c_matrix)

    evaluation.append(c_matrix)
    
