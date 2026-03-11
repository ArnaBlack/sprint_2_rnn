# замер метрик lstm модели
from rouge_score import rouge_scorer
import numpy as np
import torch

from .constants import BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN

def calculate_rouge(model, data_loader, idx2word, word2idx, device, num_samples=50):
    """
    Считает ROUGE-1 и ROUGE-2 на валидационной/тестовой выборке
    Сценарий: Модель получает 75% текста, пытается сгенерировать оставшиеся 25%
    """
    
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    
    rouge1_f1 = []
    rouge2_f1 = []
    samples_count = 0
    
    special_token_indices = {
        word2idx.get(PAD_TOKEN, 0),
        word2idx.get(UNK_TOKEN, 1),
        word2idx.get(BOS_TOKEN, 2),
        word2idx.get(EOS_TOKEN, 3)
    }
    
    with torch.no_grad():
        for batch in data_loader:
            if samples_count >= num_samples:
                break
                
            inputs = batch['input'].to(device)
            lengths = batch['length']  # Реальные длины последовательностей
            batch_size, max_seq_len = inputs.shape
            
            for i in range(batch_size):
                if samples_count >= num_samples:
                    break
                
                # Получаем реальную длину без padding
                real_len = lengths[i].item()
                
                # Пропускаем слишком короткие тексты
                if real_len < 8:
                    continue
                
                # Разделяем на 75% / 25% от РЕАЛЬНОЙ длины
                split_point = int(real_len * 0.75)
                if split_point < 6:
                    continue
                
                # Берем только реальные токены (без padding)
                input_part = inputs[i, :split_point].tolist()
                target_part = inputs[i, split_point:real_len].tolist()
                
                if len(target_part) == 0:
                    continue
                
                # Генерация
                generated_seq = []
                current_tokens = input_part.copy()
                
                for _ in range(len(target_part)):
                    context = current_tokens[-10:]
                    x = torch.tensor([context], device=device)
                    out, _ = model(x)
                    next_token = out[0, -1, :].argmax().item()
                    generated_seq.append(next_token)
                    current_tokens.append(next_token)
                
                # Конветрация в текст
                target_text = ' '.join([idx2word.get(idx, UNK_TOKEN) 
                                      for idx in target_part 
                                      if idx not in special_token_indices])
                
                generated_text = ' '.join([idx2word.get(idx, UNK_TOKEN) 
                                         for idx in generated_seq 
                                         if idx not in special_token_indices])
                
                # Метрики
                if len(target_text.strip()) > 2 and len(generated_text.strip()) > 2:
                    scores = scorer.score(target_text, generated_text)
                    rouge1_f1.append(scores['rouge1'].fmeasure)
                    rouge2_f1.append(scores['rouge2'].fmeasure)
                    samples_count += 1
    
    avg_rouge1 = sum(rouge1_f1) / len(rouge1_f1) if rouge1_f1 else 0
    avg_rouge2 = sum(rouge2_f1) / len(rouge2_f1) if rouge2_f1 else 0
    
    return avg_rouge1, avg_rouge2