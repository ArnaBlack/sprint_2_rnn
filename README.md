# Sprint 2: RNN Language Model

Проект по разработке и обучению LSTM-модели для генерации текста на основе датасета Sentiment140.

## 📋 Описание

Данный проект представляет собой языковую модель на основе LSTM (Long Short-Term Memory), которая обучается предсказывать следующий токен в последовательности. Модель способна генерировать новые тексты на основе начальной фразы (seed).

### Основные возможности

- **Предобработка данных**: очистка текста, токенизация, создание последовательностей
- **Обучение LSTM модели**: с использованием градиентного клиппинга и early stopping
- **Генерация текста**: продолжение начальной фразы с использованием сэмплирования
- **Оценка качества**: метрики ROUGE-1 и ROUGE-2 для оценки сгенерированного текста

## 📁 Структура проекта

```
sprint_2_rnn/
├── data/
│   └── tokenized_texts.pkl      # Токенизированные данные (сохраняются автоматически)
├── models/                       # Директория для сохранения моделей
├── src/
│   ├── constants.py             # Константы проекта (токены, гиперпараметры)
│   ├── data_utils.py            # Утилиты для загрузки и обработки данных
│   ├── early_stopping.py        # Реализация early stopping
│   ├── eval_lstm.py             # Функция оценки модели (loss + ROUGE)
│   ├── eval_transformer_pipeline.py  # Оценка Transformer модели (DistilGPT2)
│   ├── eval_utils.py            # Общие утилиты для оценки (подготовка выборок)
│   ├── generate_examples_lstm.py # Генерация примеров текста
│   ├── lstm_model.py            # Архитектура LSTM модели
│   ├── lstm_test.py             # Тестирование модели
│   ├── lstm_train.py            # Обучение модели
│   ├── next_token_dataset.py    # PyTorch Dataset для next token prediction
│   └── rouge_scores_lstm.py     # Расчёт ROUGE метрик для LSTM
├── requirements_sprint_2.txt    # Зависимости проекта
├── solution.ipynb               # Jupyter notebook с решением
└── README.md                    # Документация
```

## 🔧 Компоненты

### `constants.py`
Определяет основные константы проекта:
- Специальные токены: `<BOS>`, `<EOS>`, `<PAD>`, `<UNK>`
- Гиперпараметры: `RANDOM_SEED=42`, `TEST_SIZE=0.2`, `SEQ_LEN=64`, `BATCH_SIZE=128`

### `data_utils.py`
Функции для работы с данными:
- `clean_text(text)` - очистка от ссылок, упоминаний, эмодзи
- `load_and_clear_data()` - загрузка датасета Sentiment140 (10,000 текстов)
- `tokenize_texts(texts)` - токенизация с добавлением BOS/EOS токенов
- `save_tokenized()` / `load_or_tokenize()` - сохранение/загрузка токенизированных данных
- `create_sequences()` - создание пар (X, Y) для обучения next token prediction
- `create_data_split()` - разделение на train/val/test (80%/10%/10%)
- `build_vocab()` - построение словаря токенов

### `lstm_model.py`
Класс `LSTMLanguageModel`:
- Embedding слой (256 dim)
- 2 слоя LSTM (128 hidden units)
- Dropout (0.3)
- Linear слой для предсказания следующего токена
- Метод `generate()` для генерации текста с temperature sampling

### `next_token_dataset.py`
Класс `NextTokenDataset` (наследует `torch.utils.data.Dataset`):
- Преобразование токенов в индексы
- Padding до максимальной длины
- Создание attention mask
- Возвращает dict: `{input, label, attention_mask, length}`

### `lstm_train.py`
Функция `train_model()`:
- Обучение с прогресс-баром (tqdm)
- Градиентный клиппинг (`clip_grad_norm_`)
- Валидация с расчётом ROUGE метрик
- Early stopping с восстановлением лучших весов

### `early_stopping.py`
Класс `EarlyStopping`:
- Патченс (по умолчанию 3 эпохи)
- Минимальное улучшение (min_delta=0.001)
- Восстановление лучших весов

### `eval_lstm.py`
Функция `eval_lstm()`:
- Расчёт среднего loss на валидационной/тестовой выборке
- Вычисление ROUGE-1 и ROUGE-2 метрик

### `eval_utils.py`
Общие утилиты для оценки моделей (LSTM и Transformer):
- `prepare_generation_sample()` - подготовка выборки: удаление паддинга, разделение 75%/25%, конвертация в текст
- `filter_special_tokens()` - фильтрация специальных токенов (PAD, UNK, BOS, EOS)

### `rouge_scores_lstm.py`
Функция `calculate_rouge()`:
- Сценарий: модель получает 75% текста, генерирует оставшиеся 25%
- Расчёт ROUGE-1 и ROUGE-2 через `rouge_score.RougeScorer`
- Фильтрация специальных токенов

### `lstm_test.py`
Функция `test_model()`:
- Финальная оценка на тестовой выборке (10%)
- Вывод Test Loss, ROUGE-1, ROUGE-2

### `generate_examples_lstm.py`
Функция `generate_examples()`:
- Генерация текста для заданных seed-фраз
- Temperature sampling (по умолчанию 0.8)
- Фильтрация специальных токенов из вывода

## 🚀 Установка и запуск

### 1. Установка зависимостей

```bash
pip install -r requirements_sprint_2.txt
```

### 2. Запуск обучения

Пример использования в Jupyter notebook или Python скрипте:

```python
from src.data_utils import (
    load_and_clear_data, 
    load_or_tokenize, 
    create_sequences, 
    create_data_split, 
    build_vocab
)
from src.next_token_dataset import NextTokenDataset
from src.lstm_model import LSTMLanguageModel
from src.lstm_train import train_model
from src.lstm_test import test_model
from src.generate_examples_lstm import generate_examples
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

# Константы
from src.constants import BATCH_SIZE, SEQ_LEN

# 1. Загрузка и подготовка данных
texts = load_and_clear_data()
tokenized = load_or_tokenize(texts)
X, Y = create_sequences(tokenized)
X_train, Y_train, X_val, Y_val, X_test, Y_test = create_data_split(X, Y)

# 2. Построение словаря
word2idx, idx2word = build_vocab(tokenized)
vocab_size = len(word2idx)

# 3. Создание DataLoader
train_dataset = NextTokenDataset(X_train, Y_train, word2idx)
val_dataset = NextTokenDataset(X_val, Y_val, word2idx)
test_dataset = NextTokenDataset(X_test, Y_test, word2idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 4. Инициализация модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMLanguageModel(
    vocab_size=vocab_size,
    word2idx=word2idx,
    idx2word=idx2word,
    embedding_dim=256,
    hidden_dim=128,
    num_layers=2,
    dropout=0.3
).to(device)

# 5. Обучение
loss_fn = CrossEntropyLoss(ignore_index=0)  # 0 = PAD_TOKEN
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses, val_losses = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    idx2word=idx2word,
    word2idx=word2idx,
    num_epochs=10,
    device=device,
    early_stopping_patience=3
)

# 6. Тестирование
test_model(model, test_loader, idx2word, word2idx, device, loss_fn)

# 7. Генерация примеров
seed_texts = ["i love", "i hate", "this is", "the best"]
generate_examples(model, seed_texts, word2idx, max_length=20, temperature=0.8, device=device)
```

## 📊 Метрики

Модель оценивается по следующим метрикам:

- **Loss** - Cross-Entropy loss между предсказанными и истинными токенами
- **ROUGE-1** - F1 мера пересечения unigram между сгенерированным и целевым текстом
- **ROUGE-2** - F1 мера пересечения bigram между сгенерированным и целевым текстом

## ⚙️ Гиперпараметры

| Параметр | Значение |
|----------|----------|
| Размер словаря | ~10,000 (зависит от данных) |
| Embedding dim | 256 |
| Hidden dim | 128 |
| Число слоёв LSTM | 2 |
| Dropout | 0.3 |
| Batch size | 128 |
| Макс. длина последовательности | 64 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Early stopping patience | 3 |

## 📝 Примечания

- Датасет Sentiment140 загружается через `datasets.load_dataset()`
- Первые 10,000 текстов используются для обучения
- Токенизированные данные сохраняются в `data/tokenized_texts.pkl` для ускорения последующих запусков
- Модель использует teacher forcing во время обучения
- Во время генерации применяется temperature sampling для разнообразия выходных текстов

## 🎯 Примеры генерации

После обучения модель может генерировать тексты на основе начальной фразы:

```
🌱 Seed: "i love"
🤖 Output: this movie is so good and i love it

🌱 Seed: "i hate"
🤖 Output: this film it is not funny at all

🌱 Seed: "this is"
🤖 Output: a must see for anyone who loves movies
```

## 📚 Зависимости

Основные библиотеки:
- `torch` - фреймворк для глубокого обучения
- `datasets` - загрузка датасетов
- `nltk` - токенизация текста
- `rouge_score` - расчёт ROUGE метрик
- `scikit-learn` - разделение данных
- `tqdm` - прогресс-бары

Полный список зависимостей в `requirements_sprint_2.txt`.
