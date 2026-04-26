from torch.utils.data import Dataset, Subset
import torch
from sklearn.model_selection import train_test_split
import numpy as np


# Special class for working with PyTorch datasets.
# It is used by create_dataset and load_dataset methods to wrap the data.
class ManipulationDataset(Dataset):
    def __init__(self, data, labels, floatlabels=False, transform=None):
        self.data = data
        self.floatlabels = floatlabels

        if self.floatlabels:
            self.labels = np.array(labels, dtype=np.float32)
        else:
            self.labels = labels

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx], self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# A method to save dataset into file.
def create_dataset(dataset, name=None):
    data = [dataset[i][0] for i in range(len(dataset))]
    labels = [dataset[i][1] for i in range(len(dataset))]

    dataset = ManipulationDataset(data, labels)

    torch.save({'data': dataset.data, 'labels': dataset.labels}, f'{name}.pth')


# A method to load dataset from file.
# A special flag floatlabels is defined to convert labels to float32 for the regression task.
def load_dataset(path, transform=None, floatlabels=None):
    dataset = torch.load(path)

    data = dataset['data']
    labels = dataset['labels']

    return ManipulationDataset(data, labels, transform=transform, floatlabels=floatlabels)


# A train-test split method designed to split PyTorch datasets into training and testing sets.
def split(data, test_size=0.2, random_state=42):
    indices = list(range(len(data)))
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)

    return Subset(data, train_indices), Subset(data, test_indices)
