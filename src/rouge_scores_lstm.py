# замер метрик lstm модели
from rouge_score import rouge_scorer
import torch

from .constants import ROUGE_SCORES_SAMPLES, UNK_TOKEN
from .eval_utils import prepare_generation_sample, filter_special_tokens


def calculate_rouge(model, data_loader, idx2word, word2idx, device, need_print_generated_texts=False, num_samples=ROUGE_SCORES_SAMPLES):
    """
    Считает ROUGE-1 и ROUGE-2 на валидационной/тестовой выборке
    Сценарий: Модель получает 75% текста, пытается сгенерировать оставшиеся 25%
    """

    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)

    rouge1_f1 = []
    rouge2_f1 = []
    examples_shown = 0

    with torch.no_grad():
        for batch in data_loader:
            if examples_shown >= num_samples:
                break

            inputs = batch['input'].to(device)
            batch_size = inputs.shape[0]

            for i in range(batch_size):
                if examples_shown >= num_samples:
                    break

                # Подготовка выборки (удаление паддинга, разделение 75%/25%)
                sample = prepare_generation_sample(inputs[i], idx2word)
                
                if sample is None:
                    continue
                
                start_tokens = sample['start_tokens']
                target_tokens = sample['target_tokens']

                # Генерация
                generated_full_text, generated_last_part_text = model.generate(start_tokens, len(target_tokens))

                # Фильтрация специальных токенов и конвертация в текст
                target_tokens_filtered = filter_special_tokens(target_tokens, word2idx)
                
                input_text = ' '.join([idx2word.get(idx, UNK_TOKEN) for idx in start_tokens])
                target_text = ' '.join([idx2word.get(idx, UNK_TOKEN) for idx in target_tokens_filtered])


                if (need_print_generated_texts):
                    if examples_shown == 0:
                        print("\n" + "="*60)
                        print("🥑Примеры генерации модели (LSTM)")
                        print("="*60)

                    print(f"\n[Пример {examples_shown + 1}]")
                    print(f"Вход (75%):     {input_text}")
                    print(f"Цель (25%):     {target_text}")
                    print(f"Предсказание:   {generated_last_part_text}")
                    print(f"Полное предсказание:   {generated_full_text}")
                    print("="*60)  


                # Метрики
                scores = scorer.score(target_text, generated_full_text)
                rouge1_f1.append(scores['rouge1'].fmeasure)
                rouge2_f1.append(scores['rouge2'].fmeasure)
                examples_shown += 1

    avg_rouge1 = sum(rouge1_f1) / len(rouge1_f1) if rouge1_f1 else 0
    avg_rouge2 = sum(rouge2_f1) / len(rouge2_f1) if rouge2_f1 else 0

    return avg_rouge1, avg_rouge2