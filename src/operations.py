from src import files, cnn, knc

def create_data_set():
    data = files.read('./ClsKLData/kneeKL224/train')
    files.create_images(data, './ClsKLData/kneeKL224/')

MAX=10000000

def train_validate(rounds_=5, end_layer_=5, amount_=MAX):
    binary_ = False if end_layer_ == 5 else True
    data_set_, data_set_label_ = files.read_data_set('./ClsKLData/kneeKL224/data_set/', amount=amount_, binary=binary_)

    cnn.create_model(
        data_set=data_set_,
        data_set_label=data_set_label_,
        end_layer=end_layer_,
        path=f'./ClsKLData/kneeKL224/model/{end_layer_}',
        rounds=rounds_
    )

def train_model(model, rounds_=5, end_layer_=5, amount_=MAX):
    binary_ = False if end_layer_ == 5 else True
    data_set_, data_set_label_ = files.read_data_set('./ClsKLData/kneeKL224/data_set/', amount=amount_, binary=binary_)

    cnn.train_model(model, data_set_, data_set_label_, f'./ClsKLData/kneeKL224/model/{end_layer_}', end_layer_, rounds_)

def predict_five_class(model):
    cnn.evaluete_model(model, 5)

def predict_binary(model):
    cnn.evaluete_model(model, 2)

def knn_binary():
    knc.train_validate(True) 

def knn_five_class():
    knc.train_validate(False) 

def create_data_set_val(path_origin, path_destiny):
    val = files.read_val(path_origin)
    files.create_images_val(val, path_destiny)
