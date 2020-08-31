
import numpy as np
from sklearn.model_selection import train_test_split

def split_data_into_train_and_test(dataset_path, dataset_name, dataset_file_extension):    
    dataset_path_and_name = dataset_path + dataset_name
    data_file = dataset_path_and_name + "." + dataset_file_extension
    train_file = dataset_path_and_name + ".train"
    test_file = dataset_path_and_name + ".test"
    
    #read data file
    with open(data_file, "r") as f:
        data = f.readlines()
    
    file_length = len(data)
    lines = np.arange(file_length)
    train, test = train_test_split(lines, test_size=0.2)
    
    #create train file
    with open(train_file, "a") as f:
        for line in train:
            f.write(data[line]) 
    
    
    #create test file
    with open(test_file, "a") as f:
        for line in test:
            f.write(data[line])      


def main():
    #split the file /arch-forest/data/adult/adult.data into adult.train and adult.test
    dataset_path = "/arch-forest/data/adult/"
    dataset_name = "adult"
    dataset_file_extension = "data"
    split_data_into_train_and_test(dataset_path, dataset_name, dataset_file_extension)
