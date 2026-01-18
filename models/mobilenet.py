import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class MobileNetV2Med(nn.Module):
    """
    MobileNet V2 (1 channel input, 6 classes)
    """
    def __init__(self, num_classes=6):
        super(MobileNetV2Med, self).__init__()
        # Pretrained weights는 사용하지 않음 (구조만 로드)
        self.model = models.mobilenet_v2(weights=None)

        # 1채널 입력(Grayscale)으로 수정
        # MobileNetV2의 첫 레이어는 features[0][0]
        # 기본: nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # 마지막 분류 레이어 수정
        # 기본: nn.Linear(1280, 1000)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        return self.model(x)

def load_model(model_path, device):
    """모델 가중치 로드"""
    model = MobileNetV2Med(num_classes=6)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Warning: Failed to load weights directly: {e}")
        print("Returning initialized model without weights for demo purposes.")

    model.to(device)
    model.eval()
    return model

def preprocess_image(image: Image.Image):
    """이미지 전처리"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image).unsqueeze(0) # Add batch dimension

def predict(model, image, device, class_names=None):
    """추론 수행"""
    if class_names is None:
        # Default Medical Classes
        class_names = ['AbdomenCT', 'BreastMRI', 'CXR', 'ChestCT', 'Hand', 'HeadCT']

    image_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_idx].item()

    result = {
        "predicted_class": class_names[predicted_idx],
        "confidence": confidence,
        "probabilities": {class_names[i]: prob.item() for i, prob in enumerate(probabilities[0])}
    }
    return result

