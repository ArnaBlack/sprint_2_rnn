from .eval_lstm import eval_lstm

def test_model(model, test_loader, idx2word, word2idx, device, loss_fn):
    avg_test_loss, rouge1, rouge2 = eval_lstm(model, test_loader, idx2word, word2idx, device, loss_fn)

    print("\n" + "="*60)
    print("ФИНАЛЬНАЯ ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ (10%)")
    print("="*60)
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test ROUGE-1: {rouge1:.4f}, Test ROUGE-2: {rouge2:.4f} ")
    print("="*60)