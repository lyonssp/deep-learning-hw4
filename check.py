from homework.models import MLPPlanner, TransformerPlanner, CNNPlanner, calculate_model_size_mb
import torch

if __name__ == '__main__':
  mlp = MLPPlanner()
  mlp_size = calculate_model_size_mb(mlp)
  print(f"Classifier model size: {mlp_size:.2f} MB")
  print(mlp)

  transformer = TransformerPlanner()
  transformer_size = calculate_model_size_mb(transformer)
  print(f"Detector model size: {transformer_size:.2f} MB")
  print(transformer)
  
  cnn = CNNPlanner()
  cnn_size = calculate_model_size_mb(cnn)
  print(f"Detector model size: {cnn_size:.2f} MB")
  print(cnn)

  x = torch.randn(128, 3, 96, 128)
  y = mlp.forward(x)
  print(f"logits output shape: {y[0].shape}")
  print(f"depth output shape: {y[1].shape}")
