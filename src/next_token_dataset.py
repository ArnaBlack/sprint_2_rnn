# код с torch Dataset'ом 

import torch
from torch.utils.data import Dataset

from .constants import PAD_TOKEN, UNK_TOKEN, SEQ_LEN

class NextTokenDataset(Dataset):
    def __init__(self, X, Y, word2idx, max_len=SEQ_LEN):
        self.X = X
        self.Y = Y
        self.word2idx = word2idx
        self.max_len = max_len
        self.pad_idx = word2idx[PAD_TOKEN]
        self.unk_idx = word2idx[UNK_TOKEN]
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x_tokens = self.X[idx][:self.max_len]
        y_tokens = self.Y[idx][:self.max_len]
        
        # Преобразование в индексы
        x_indices = [self.word2idx.get(t, self.unk_idx) for t in x_tokens]
        y_indices = [self.word2idx.get(t, self.unk_idx) for t in y_tokens]

        real_x_length = len(x_tokens)
        real_y_length = len(y_tokens)
        
        # Padding до max_len
        x_padded = x_indices + [self.pad_idx] * (self.max_len - real_x_length)
        y_padded = y_indices + [self.pad_idx] * (self.max_len - real_y_length)
        # Маска внимания (1 для реальных токенов, 0 для padding)
        attention_mask = [1] * real_x_length + [0] * (self.max_len - real_x_length)
        
        return {
            'input': torch.tensor(x_padded, dtype=torch.long),
            'label': torch.tensor(y_padded, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float32),
            'length': real_x_length
        }