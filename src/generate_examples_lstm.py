from .constants import BOS_TOKEN, UNK_TOKEN, EOS_TOKEN, PAD_TOKEN
import torch

def generate_examples(model, seed_texts, word2idx, max_length=20, temperature=0.8, device=None):
    """
    Генерирует примеры текста для заданных начальных фраз.
    """
    
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    print("\n" + "="*60)
    print("ПРИМЕРЫ ГЕНЕРАЦИИ МОДЕЛИ")
    print("="*60)
    
    # Получаем индексы специальных токенов из словаря
    bos_idx = word2idx.get(BOS_TOKEN, 2)
    eos_idx = word2idx.get(EOS_TOKEN, 3)
    unk_idx = word2idx.get(UNK_TOKEN, 1)
    
    with torch.no_grad():
        for seed_text in seed_texts:
            # Токенизация начального текста
            tokens = seed_text.lower().split()
            start_indices = [bos_idx] + [word2idx.get(token, unk_idx) for token in tokens]            
            
            # Генерация продолжения
            generated_text = model.generate(
                start_tokens=start_indices, 
                max_length=max_length, 
                temperature=temperature
            )

            filtered_words = [word for word in generated_text.split() 
                            if word not in [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN]]
            filtered_text = ' '.join(filtered_words)
            
            # Вывод результата
            print(f"\n🌱 Seed: \"{seed_text}\"")
            print(f"🤖 Output: {filtered_text}")
            print("-" * 60)
    
    print("="*60)