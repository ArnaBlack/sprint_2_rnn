 # Обработка датасета
import re
from datasets import load_dataset
import nltk
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import os
import pickle
from sklearn.model_selection import train_test_split
from collections import Counter

from .constants import RANDOM_SEED, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN, TEST_SIZE, SEQ_LEN

DATA_DIR = 'data'
TOKENIZED_FILE = os.path.join(DATA_DIR, 'tokenized_texts.pkl')


# Очистка текста
def clean_text(text):
    '''Очищает текст от ссылок, упоминаний и эмодзи, а также приводит его к нижнему регистру'''
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление ссылок
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Удаление упоминаний
    text = re.sub(r'@\w+', '', text)
    # Удаление одиночных дефисов (оставляем только дефисы между буквами)
    text = re.sub(r'(?<!\w)-|-(?!\w)', '', text)
    # Удаление эмодзи (оставляем только латинские буквы, апостроф и дефисы в составе слов)
    text = re.sub(r"[^\w\s.,!?']", '', text)
    # Замена множественных пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_clear_data():
    # Загрузка датасета
    dataset = load_dataset("sentiment140", trust_remote_code=True)
    texts = dataset['train']['text']
    #[:700000]  # Используем подмножество изза нехватки памятм
    
    cleaned_texts = [clean_text(text) for text in texts if len(clean_text(text)) > 0]

    if len(cleaned_texts) % 2 != 0:
        cleaned_texts = cleaned_texts[:-1]

    print(f"✅ Загружено {len(texts)} текстов, после очистки осталось {len(cleaned_texts)} текстов.")
    print(f"Примеры очищенных текстов: {cleaned_texts[:5]}")
    return cleaned_texts


# Токенизация
def tokenize_texts(texts):
    """Токенизирует тексты, добавляя специальные токены BOS и EOS"""
    nltk.download('punkt_tab', quiet=True)

    tokenized_result = []
    for text in texts: 
        tokens = [BOS_TOKEN] + word_tokenize(text) + [EOS_TOKEN]
        tokenized_result.append(tokens)

    return tokenized_result



# Сохраняет токенизированные данные в файл
def save_tokenized(tokenized, filepath=TOKENIZED_FILE):
    """Сохраняет токенизированные данные в файл"""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(tokenized, f)
    print(f"✅ Токенизированные данные сохранены в {filepath}")


# Загружает токенизированные данные из файла или выполняет токенизацию, если файл не найден
def load_or_tokenize(texts, filepath=TOKENIZED_FILE):
    """Загружает токенизированные данные из файла или выполняет токенизацию, если файл не найден"""
    if os.path.exists(filepath):
        print(f"Загрузка токенизированных данных из {filepath}...")
        with open(filepath, 'rb') as f:
            tokenized = pickle.load(f)
    else:
        print("Токенизация текстов...")
        tokenized = tokenize_texts(texts)
        save_tokenized(tokenized, filepath)
    return tokenized    

# Формирование обучающих последовательностей
def create_sequences(tokenized_texts):
    """
    Создает пары X, Y одинаковой длины.
    Работает с предложениями любой длины.
    """
    X, Y = [], []
    
    for tokens in tokenized_texts:
        # Пропускаем слишком короткие (нужно минимум 2 токена для пары)
        if len(tokens) < 2:
            continue
            
        x_seq = tokens[:-1]
        y_seq = tokens[1:]
            
        # Контрольная проверка
        assert len(x_seq) == len(y_seq), f"✅ Длины не совпадают: {len(x_seq)} vs {len(y_seq)}"
            
        X.append(x_seq)
        Y.append(y_seq)
            
    return X, Y

# Разделение на train/val/test
def create_data_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_SEED):
    """Разделяет данные на обучающую, валидационную и тестовую выборки"""
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=random_state)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")  

    return X_train, Y_train, X_val, Y_val, X_test, Y_test



# Создание словаря
def build_vocab(tokenized_texts, max_vocab_size=20000, min_freq=2):
    """
    Создает словарь только из частых токенов.
    
    max_vocab_size: Максимальный размер словаря (критично для памяти)
    min_freq: Минимальная частота токена для включения в словарь
    """
    # Считаем частоту каждого токена
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)
    
    print(f"Всего уникальных токенов: {len(counter)}")
    print(f"Топ-10 частых: {counter.most_common(10)}")
    
    # Создаем словарь со специальными токенами
    word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1, BOS_TOKEN: 2, EOS_TOKEN: 3}
    idx2word = {0: PAD_TOKEN, 1: UNK_TOKEN, 2: BOS_TOKEN, 3: EOS_TOKEN}
    
    # Добавляем только самые частые токены
    # -4 потому что у нас уже 4 специальных токена
    most_common = counter.most_common(max_vocab_size - 4)
    
    for token, count in most_common:
        if count >= min_freq:  # Пропускаем очень редкие слова
            idx = len(word2idx)
            word2idx[token] = idx
            idx2word[idx] = token
    
    print(f"✅ Размер словаря: {len(word2idx)} (ограничено с {len(counter)})")
    return word2idx, idx2word