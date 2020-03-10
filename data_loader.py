import numpy as np
import pickle
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import MinMaxScaler 


def load_dataset(data_dir, fold_count):
    print('load_dataset')
    
    filename = data_dir + '/fold_' + fold_count
    print(filename)
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    train_y = data['train']['y'].reshape(-1)
    # print('train_y', train_y)
    train_y[train_y<0] = 0
    # train_y[train_y>0] = 1
    # train_y = train_y.squeeze(-1)
    index = np.arange(len(train_y))
    # np.random.shuffle(index)
    train_graph = data['train']['X'][index]
    train_y = train_y[index]
    train_label = np.zeros((len(data['train']['X']), 2))


    test_y = data['test']['y'].reshape(-1)
    test_y[test_y<0] = 0
    index = np.arange(len(test_y))
    
    test_y = test_y[index]
    test_graph = data['test']['X']
    

    (n_graph, hw) = train_graph.shape
    n_H = int(np.sqrt(float(hw)))
    test_graph = np.array(test_graph)

    if 'DTI' in data_dir:
        scaler = MinMaxScaler()
        train_graph = scaler.fit_transform(train_graph)
        test_graph = scaler.fit_transform(test_graph)
        
    train_graph = train_graph.reshape(n_graph, 1, n_H, n_H)
    test_graph = test_graph.reshape(-1, 1, n_H, n_H)

    train_data = []
    test_data = []

    for i in range(len(train_graph)):
        train_data.append((torch.from_numpy(train_graph[i]).float(), torch.from_numpy(np.array(train_y[i])).long()))
    for i in range(len(test_graph)):
        test_data.append((torch.from_numpy(test_graph[i]).float(), torch.from_numpy(np.array(test_y[i])).long()))
    return train_data, test_data

 


def get_train_valid_loader(train_data, batch_size, random_seed, valid_size=0.1, shuffle=True, show_sample=False, num_workers=4, pin_memory=False):
  
   
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)




def get_test_loader(test_data,
                    batch_size,
                    num_workers=4,
                    pin_memory=False):
  
    batch_size = batch_size
    data_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader

