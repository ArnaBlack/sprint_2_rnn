import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001, restore_best_weights=True):
        """
        patience: сколько эпох ждать улучшения перед остановкой
        min_delta: минимальное изменение лосса, чтобы считаться улучшением
        restore_best_weights: восстановить веса лучшей модели в конце
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = None
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_loss, model):
        # Если это первая эпоха или лосс улучшился
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                # Сохраняем копию весов модели
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            print(f"  ⚠️ EarlyStopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                print("  🛑 EarlyStopping: Остановка обучения")
                if self.restore_best_weights and self.best_weights:
                    # Восстанавливаем лучшие веса
                    model.load_state_dict(self.best_weights)
                    print("  ✅ Восстановлены веса лучшей эпохи")