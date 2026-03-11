import torch
from torch.nn.utils import clip_grad_norm_
from .early_stopping import EarlyStopping
from .eval_lstm import eval_lstm

# Для красивого прогресс-бара
from tqdm import tqdm

def train_model(model, train_loader, val_loader, loss_fn, optimizer, idx2word, word2idx, num_epochs=10, device=None, early_stopping_patience=3):
    if device is None:
        device = next(model.parameters()).device

    early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=0.001, restore_best_weights=True)    
    
    train_losses = []
    val_losses = []
    rouge_scores = []
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')

        for batch in pbar:
            inputs = batch['input'].to(device)  
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Модель возвращает (output, hidden)
            outputs, _ = model(inputs)
            
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0) # Градиентный клиппинг для LSTM
            optimizer.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Валидация и ROUGE-метрики
        avg_val_loss, rouge1, rouge2 = eval_lstm(model, val_loader, idx2word, word2idx, device, loss_fn)

        val_losses.append(avg_val_loss)

        rouge_scores.append({'rouge1': rouge1, 'rouge2': rouge2})

        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}")
        print(f"Val ROUGE-1: {rouge1:.4f}, Val ROUGE-2: {rouge2:.4f} ") 
        print("="*60)
        
        early_stopping(avg_val_loss, model)
    
        if early_stopping.early_stop:
            break  # Прерываем цикл эпох    
     
    return train_losses, val_losses