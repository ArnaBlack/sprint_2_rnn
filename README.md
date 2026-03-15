# Sprint 2: RNN Language Model

Проект по разработке и обучению LSTM-модели и предобученного трансформера DistilGPT2 для генерации текста на основе датасета Sentiment140.

## Структура проекта

```
sprint_2_rnn/
├── data/
│   └── tokenized_texts.pkl      # Токенизированные данные (сохраняются автоматически)
├── models/
│   └── lstm_model_weights.pth   # Сохранённые веса LSTM модели
├── src/
│   ├── constants.py             # Константы проекта (токены, гиперпараметры)
│   ├── data_utils.py            # Загрузка, очистка, токенизация данных, создание последовательностей
│   ├── early_stopping.py        # Реализация early stopping с восстановлением лучших весов
│   ├── eval_lstm.py             # Оценка LSTM модели (loss + ROUGE)
│   ├── eval_transformer_pipeline.py  # Оценка Transformer модели (DistilGPT2)
│   ├── eval_utils.py            # Общие утилиты для оценки (подготовка выборок, фильтрация токенов)
│   ├── lstm_model.py            # Архитектура LSTM модели с методом генерации текста
│   ├── lstm_test.py             # Финальное тестирование модели на тестовой выборке
│   ├── lstm_train.py            # Обучение модели с градиентным клиппингом и прогресс-баром
│   ├── lstm_utils.py            # Утилиты для сохранения весов модели
│   ├── next_token_dataset.py    # PyTorch Dataset и collate_fn для next token prediction
│   └── rouge_scores_lstm.py     # Расчёт ROUGE-1 и ROUGE-2 метрик для LSTM
├── requirements.txt             # Зависимости проекта
├── solution.ipynb               # Jupyter notebook с решением
└── README.md                    # Документация
```

## Компоненты

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
- Embedding слой (128 dim)
- 2 слоя LSTM (128 hidden units)
- Dropout (0.3)
- Linear слой для предсказания следующего токена
- Метод `generate()` для генерации текста

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

### `lstm_utils.py`
Утилиты для работы с моделью:
- `save_model_weight(model)` - сохранение весов модели в `models/lstm_model_weights.pth`

### `eval_transformer_pipeline.py`
Оценка Transformer модели (DistilGPT2):
- Класс `DistilGPT2Model` - обёртка для модели с методом `generate()`
- Функция `test_transformer()` - финальная оценка на тестовой выборке
- Функция `evaluate_transformer()` - оценка на валидационной выборке
