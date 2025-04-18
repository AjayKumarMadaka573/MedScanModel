import torch
import torch.nn as nn
from torchvision import models
import sys

# Load your existing model class
class Classifier(nn.Module):
    def __init__(self, backbone, num_classes=2, bottleneck_dim=512):
        super().__init__()
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features = self.bottleneck(x)
        return self.classifier(features), features

def convert(pth_path, onnx_path):
    # 1. Initialize model
    backbone = models.resnet18(weights=None)
    model = Classifier(backbone)
    
    # 2. Load weights
    checkpoint = torch.load(pth_path, map_location='cpu')
    if any(k.startswith('module.') for k in checkpoint.keys()):
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    # 3. Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 4. Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output', 'features'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
            'features': {0: 'batch_size'}
        }
    )
    print(f"Successfully converted to {onnx_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_onnx.py <input.pth> <output.onnx>")
        sys.exit(1)
    
    convert(sys.argv[1], sys.argv[2])