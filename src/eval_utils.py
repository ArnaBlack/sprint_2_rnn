# Утилиты для оценки моделей (LSTM и Transformer)
import torch
from .constants import PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN


def prepare_generation_sample(seq, idx2word, word2idx, split_ratio=0.75, min_seq_len=4, min_target_len=1):
    """
    Подготавливает выборку для генерации текста:
    1. Удаляет паддинг
    2. Проверяет минимальную длину
    3. Разделяет на input (75%) и target (25%)
    4. Конвертирует индексы в текст
    """
    # Удаляем паддинг (индекс 0)
    seq = seq[seq != 0]
    seq_len = len(seq)
    
    # Проверка минимальной длины
    if seq_len < min_seq_len:
        return None
    
    # Разделение на 75% / 25%
    split_idx = int(seq_len * split_ratio)
    
    # Конвертация индексов в токены
    start_tokens = seq[:split_idx].tolist()
    target_tokens = seq[split_idx:].tolist()
    
    # Проверка длины target
    if len(target_tokens) < min_target_len:
        return None
    
    # Конвертация в текст
    input_text = ' '.join([idx2word.get(idx, UNK_TOKEN) for idx in filter_special_tokens(start_tokens, word2idx)])
    target_text = ' '.join([idx2word.get(idx, UNK_TOKEN) for idx in filter_special_tokens(target_tokens, word2idx)])
    # Дополнительная проверка по количеству слов
    if len(input_text.split()) < 3 or len(target_text.split()) < 1:
        return None
    
    return {
        'input_text': input_text,
        'target_text': target_text,
        'start_tokens': start_tokens,
        'target_tokens': target_tokens
    }


def filter_special_tokens(tokens, word2idx):
    """Удаляет специальные токены (PAD, UNK, BOS, EOS) из списка индексов или текста"""
    special_token_indices = {
        word2idx[PAD_TOKEN],
        word2idx[UNK_TOKEN],
        word2idx[BOS_TOKEN],
        word2idx[EOS_TOKEN]
    }

    if isinstance(tokens, list):
        return [idx for idx in tokens if idx not in special_token_indices]
    elif isinstance(tokens, str):
        # Если строка - разбиваем на токены, фильтруем, собираем обратно
        token_list = tokens.split()
        filtered = [t for t in token_list if t not in {PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN}]
        return ' '.join(filtered)

    return tokens
