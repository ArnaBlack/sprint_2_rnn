# код с запуском и замером качества трансформера
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from evaluate import load as load_metric

from .constants import ROUGE_SCORES_SAMPLES, MAX_GENERATION_LENGTH
from .eval_utils import prepare_generation_sample

# Для красивого прогресс-бара
from tqdm import tqdm

class DistilGPT2Model:
    def __init__(self, model_name='distilgpt2', device='cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        self.model.to(self.device) 

    def generate(self, prompt, max_length=MAX_GENERATION_LENGTH, temperature=0.8):
        """
        Генерирует текст на основе промпта.

        Args:
            prompt: входной текст
            max_length: максимальное количество новых токенов для генерации
            temperature: температура сэмплирования

        Returns:
            Сгенерированный текст (только новая часть)
        """
        self.model.eval()

        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.shape[1]

        attention_mask = torch.ones_like(inputs).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated__last_part_tokens = outputs[0, input_length:]
        generated_last_part_text = self.tokenizer.decode(generated__last_part_tokens, skip_special_tokens=True)

        generated_full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_full_text.strip(), generated_last_part_text.strip()


def evaluate_transformer(val_loader, idx2word, word2idx, model_name='distilgpt2', device=None, num_examples=ROUGE_SCORES_SAMPLES, max_new_tokens=MAX_GENERATION_LENGTH):
    model = DistilGPT2Model(model_name=model_name, device=device)

    rouge_metric = load_metric("rouge")

    print("\n" + "="*60)
    print("Оценка на валидационной выборке (DISTILGPT2)")
    print("="*60)

    predictions = []
    references = []
    examples_shown = 0
    


    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Processing batches")):
            inputs = batch['input'].to(device)  # Тензор LSTM индексов

            for i in range(inputs.shape[0]):
                # Подготовка выборки (удаление паддинга, разделение 75%/25%)
                sample = prepare_generation_sample(inputs[i], idx2word, word2idx)
                
                if sample is None:
                    continue
                
                input_text = sample['input_text']
                target_text = sample['target_text']

                # Генерация (трансформер сам затокенизирует input_text)
                generated_full_text, generated_last_part_text = model.generate(input_text, max_length=len(target_text))

                predictions.append(generated_full_text)
                references.append(target_text)

                # Вывод примеров для отладки
                if examples_shown < num_examples:
                    print(f"\n[Пример {examples_shown + 1}]")
                    print(f"Вход (75%):     {input_text}")
                    print(f"Цель (25%):     {target_text}")
                    print(f"Предсказание:   {generated_last_part_text}")
                    print(f"Полное предсказание:   {generated_full_text}")
                    print("="*60)  
                    examples_shown += 1  

    print("\n" + "="*60)
    print("Метрики ROUGE (DISTILGPT2)")
    print("="*60)   

    if len(predictions) > 0:
            results = rouge_metric.compute(predictions=predictions, references=references)
            
            print(f"ROUGE-1 (F1): {results['rouge1'].mid.fmeasure:.4f}")
            print(f"ROUGE-2 (F1): {results['rouge2'].mid.fmeasure:.4f}")
            print("=" * 70)
            
            return results
    else:
        print("Нет данных для расчета метрик! Проверьте val_loader и idx2word")
        return None             