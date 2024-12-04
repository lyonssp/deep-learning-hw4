from homework.models import MLPPlanner, TransformerPlanner, CNNPlanner, calculate_model_size_mb
import torch

if __name__ == '__main__':
  batch_size = 128
  model = TransformerPlanner()
  left = torch.randn(batch_size, 10, 2)
  right = torch.randn(batch_size, 10, 2)
  preds = model.forward(left, right)
  print(preds.shape)
