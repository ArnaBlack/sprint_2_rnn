# код с torch Dataset'ом 

import torch
from torch.utils.data import Dataset

from .constants import PAD_TOKEN, UNK_TOKEN, SEQ_LEN


def collate_fn(batch):
    # Находим максимальную длину в текущем батче
    max_len = max(item['input'].size(0) for item in batch)
    
    inputs_padded = []
    labels_padded = []
    masks = []
    lengths = []
    
    for item in batch:
        seq_len = item['input'].size(0)
        
        # Паддим только до max_len в этом батче
        pad_amount = max_len - seq_len
        
        x_padded = torch.nn.functional.pad(
            item['input'], (0, pad_amount), value=0  # 0 = pad_idx
        )
        y_padded = torch.nn.functional.pad(
            item['label'], (0, pad_amount), value=0
        )
        mask = torch.nn.functional.pad(
            torch.ones(seq_len), (0, pad_amount), value=0
        )
        
        inputs_padded.append(x_padded)
        labels_padded.append(y_padded)
        masks.append(mask)
        lengths.append(seq_len)
    
    return {
        'input': torch.stack(inputs_padded),
        'label': torch.stack(labels_padded),
        'attention_mask': torch.stack(masks),
        'lengths': torch.tensor(lengths)
    }

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
    # Возвращаем "сырые" индексы, без паддинга
        x_tokens = self.X[idx][:self.max_len]  # только обрезка если нужно
        y_tokens = self.Y[idx][:self.max_len]
        
        x_indices = [self.word2idx.get(t, self.unk_idx) for t in x_tokens]
        y_indices = [self.word2idx.get(t, self.unk_idx) for t in y_tokens]
        
        return {
            'input': torch.tensor(x_indices, dtype=torch.long),  # без паддинга!
            'label': torch.tensor(y_indices, dtype=torch.long),
            'length': len(x_indices)
        }