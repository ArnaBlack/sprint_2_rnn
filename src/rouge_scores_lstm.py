# замер метрик lstm модели
from rouge_score import rouge_scorer
import torch
from tqdm import tqdm

from .constants import ROUGE_SCORES_SAMPLES, UNK_TOKEN
from .eval_utils import prepare_generation_sample


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

    pbar = tqdm(data_loader, desc="Calculating ROUGE scores")

    with torch.no_grad():
        for batch in pbar:

            inputs = batch['input'].to(device)
            batch_size = inputs.shape[0]

            for i in range(batch_size):

                # Подготовка выборки (удаление паддинга, разделение 75%/25%)
                sample = prepare_generation_sample(inputs[i], idx2word, word2idx)
                
                if sample is None:
                    continue
                
                start_tokens = sample['start_tokens']
                target_tokens = sample['target_tokens']
                input_text = sample['input_text']
                target_text = sample['target_text']

                # Генерация
                generated_full_text, generated_last_part_text = model.generate(start_tokens, len(target_tokens))


                if (need_print_generated_texts):
                    if examples_shown == 0:
                        print("\n" + "="*60)
                        print("🥑Примеры генерации модели (LSTM)")
                        print("="*60)
                    if examples_shown < num_samples:   
                        print(f"\n[Пример {examples_shown + 1}]")
                        print(f"Вход (75%): {input_text}")
                        print(f"Цель (25%): {target_text}")
                        print(f"Предсказание: {generated_last_part_text}")
                        print(f"Полное предложение: {generated_full_text}")
                        print("="*60)  


                # Метрики
                scores = scorer.score(target_text, generated_last_part_text)
                rouge1_f1.append(scores['rouge1'].fmeasure)
                rouge2_f1.append(scores['rouge2'].fmeasure)
                examples_shown += 1

    avg_rouge1 = sum(rouge1_f1) / len(rouge1_f1) if rouge1_f1 else 0
    avg_rouge2 = sum(rouge2_f1) / len(rouge2_f1) if rouge2_f1 else 0

    return avg_rouge1, avg_rouge2