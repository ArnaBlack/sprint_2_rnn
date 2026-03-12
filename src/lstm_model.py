# код lstm модели
import torch
import torch.nn as nn
from .eval_utils import filter_special_tokens

from .constants import BOS_TOKEN, EOS_TOKEN, UNK_TOKEN

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, word2idx, idx2word,  embedding_dim=256, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.word2idx = word2idx
        self.idx2word = idx2word
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden=None):

        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        lstm_out, hidden = self.lstm(embedded, hidden)
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        
        output = self.fc(lstm_out)  # (batch_size, seq_len, vocab_size)
        return output, hidden
    
    def generate(self, start_tokens, max_length=20, temperature=0.8):
        self.eval()

        device = next(self.parameters()).device

        generated_seq = []
        
        # Получаем индекс токена завершения из словаря
        eos_idx = self.word2idx[EOS_TOKEN]

        with torch.no_grad():
            current_tokens = start_tokens
            generated = start_tokens.copy()
            hidden = None

            for _ in range(max_length):
                context = current_tokens
                # Преобразование в тензор
                x = torch.tensor([context], device=device)
                output, hidden = self.forward(x, hidden)

                # Получаем предсказание для последнего токена
                logits = output[0, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)

                # Сэмплируем следующий токен
                next_token = torch.multinomial(probs, 1).item()

                if next_token == eos_idx:
                    break

                generated.append(next_token)
                current_tokens.append(next_token)
                generated_seq.append(next_token)

            # Преобразование индексов обратно в слова
            generated_full_text = ' '.join([self.idx2word.get(idx, UNK_TOKEN) for idx in filter_special_tokens(generated, self.word2idx)])
            generated_last_part_text = ' '.join([self.idx2word.get(idx, UNK_TOKEN) for idx in filter_special_tokens(generated_seq, self.word2idx)])
            return generated_full_text, generated_last_part_text