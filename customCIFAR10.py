import random
from torch.utils.data import Dataset

class CustomCIFAR10(Dataset):
    def __init__(self, dataset, noise_type=None, noise_rate=0.0):
        self.dataset = dataset
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.labels = [label for _, label in dataset]
        
        if noise_type == "random_shuffle":
            random.shuffle(self.labels)
        elif noise_type == "label_noise":
            num_noisy = int(len(self.labels) * noise_rate)
            noisy_indices = random.sample(range(len(self.labels)), num_noisy)
            for idx in noisy_indices:
                original_label = self.labels[idx]
                possible_labels = list(range(10))
                possible_labels.remove(original_label)
                self.labels[idx] = random.choice(possible_labels)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        label = self.labels[idx]
        return img, label
    