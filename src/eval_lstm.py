import torch

from .rouge_scores_lstm import calculate_rouge

def eval_lstm(model, data_loader, idx2word, word2idx, device, loss_fn):
    """Считает среднюю потерю на валидационной/тестовой выборке, а также ROUGE-1 и ROUGE-2"""
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['input'].to(device)
            labels = batch['label'].to(device)

            outputs, _ = model(inputs)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item()

    avg_val_loss = total_loss / len(data_loader)


    rouge1, rouge2 = calculate_rouge(model, data_loader, idx2word, word2idx, device, need_print_generated_texts=True)

    return avg_val_loss, rouge1, rouge2