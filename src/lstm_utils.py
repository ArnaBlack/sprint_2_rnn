import torch
import os


MODEL_WEIGHTS_DIR = 'models'

def save_model_weight(model):
    os.makedirs('models', exist_ok=True)
    modelWeightPath = os.path.join(MODEL_WEIGHTS_DIR, 'lstm_model_weights.pth')
    torch.save(model.state_dict(), modelWeightPath)
    print(f"✅ Веса модели сохранены в {modelWeightPath}")