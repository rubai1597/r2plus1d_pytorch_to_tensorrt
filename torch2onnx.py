import torch
from torch import nn
from torch import onnx
from torchvision.models.video.resnet import r2plus1d_18


if __name__ == "__main__":
    model = r2plus1d_18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.eval()

    x = torch.randn(1, 3, 8, 112, 112)

    torch.onnx.export(
        model,
        x,
        "r2plus1d_18.onnx",
        export_params=True,
    )
