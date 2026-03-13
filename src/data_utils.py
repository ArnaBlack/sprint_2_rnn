 # Обработка датасета
import re
from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize
import os
import pickle
from sklearn.model_selection import train_test_split

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
    texts = dataset['train']['text'][:10000]  # Используем подмножество для демонстрации
    
    cleaned_texts = [clean_text(text) for text in texts if len(clean_text(text)) > 0]

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
def create_sequences(tokenized_texts, max_seq_len=SEQ_LEN):
    """
    Создает пары X, Y одинаковой длины.
    Работает с предложениями любой длины.
    """
    X, Y = [], []
    
    for tokens in tokenized_texts:
        # Пропускаем слишком короткие (нужно минимум 2 токена для пары)
        if len(tokens) < 2:
            continue
            
        # Проходим по всем возможным позициям начала окна
        for i in range(len(tokens) - 1):
            # Динамически считаем доступную длину до конца текста
            # Но не больше желаемого max_seq_len
            available_len = len(tokens) - i - 1
            current_len = min(available_len, max_seq_len)
            
            if current_len < 1:
                continue
                
            # Формируем пары ОДИНАКОВОЙ длины
            x = tokens[i : i + current_len]
            y = tokens[i + 1 : i + 1 + current_len]
            
            # Контрольная проверка
            assert len(x) == len(y), f"✅ Длины не совпадают: {len(x)} vs {len(y)}"
            
            X.append(x)
            Y.append(y)
            
    return X, Y

# Разделение на train/val/test
def create_data_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_SEED):
    """Разделяет данные на обучающую, валидационную и тестовую выборки"""
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=random_state)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")  

    return X_train, Y_train, X_val, Y_val, X_test, Y_test



# Создание словаря
def build_vocab(tokenized_texts):
    """Создает словарь токенов и обратный словарь"""
    word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1, BOS_TOKEN: 2, EOS_TOKEN: 3}
    idx2word = {0: PAD_TOKEN, 1: UNK_TOKEN, 2: BOS_TOKEN, 3: EOS_TOKEN}

    for tokens in tokenized_texts:
        for token in tokens:
            if token not in word2idx:
                idx = len(word2idx)
                word2idx[token] = idx
                idx2word[idx] = token
    
    return word2idx, idx2word