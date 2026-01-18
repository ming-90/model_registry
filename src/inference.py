import torch
import os
import json
from PIL import Image
from models import mobilenet, UNet
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

class InferenceEngine:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fallback_roots = self._load_fallback_roots()

    def _load_fallback_roots(self):
        """
        환경 변수에서 FALLBACK_PATH(대체 저장소 루트)를 로드하여 리스트로 반환
        예: FALLBACK_PATH="/mnt/data, /backup"
        """
        fallback_env = os.getenv("FALLBACK_PATH")
        if not fallback_env:
            return []

        # 콤마 구분 문자열 시도
        if "," in fallback_env:
            return [p.strip() for p in fallback_env.split(",")]

        # 3. 단일 경로
        return [fallback_env]

    def _get_model_module(self, model_name: str):
        """모델 이름에 따라 적절한 모듈 반환"""
        model_name = model_name.lower()
        if "mobilenet" in model_name:
            return mobilenet
        elif "unet" in model_name:
            return UNet
        else:
            print(f"Warning: Unknown model name '{model_name}', defaulting to MobileNet.")
            return mobilenet

    def _resolve_model_path(self, original_path: str) -> str:
        """
        원래 경로가 존재하지 않을 경우, FALLBACK_PATH 루트를 사용하여 대체 경로를 탐색
        original_path: "artifacts/mobilenet/uuid/model.pth"
        """
        if os.path.exists(original_path):
            return original_path

        # artifacts/ 제거 후 상대 경로 추출 (DB에 저장된 rel_path가 보통 artifacts 하위 구조임)
        # 하지만 app.py에서는 os.path.join("artifacts", row['rel_path']) 형태로 넘겨줌.
        # 따라서 artifacts/ 접두사가 있다면 제거하고 fallback root와 결합.

        rel_path = original_path
        if original_path.startswith("artifacts/"):
            rel_path = original_path.replace("artifacts/", "", 1)
        elif original_path.startswith("artifacts\\"): # Windows 대응
            rel_path = original_path.replace("artifacts\\", "", 1)

        for root in self.fallback_roots:
            fallback_full_path = os.path.join(root, rel_path)
            if os.path.exists(fallback_full_path):
                print(f"Found model in fallback storage: {fallback_full_path}")
                return fallback_full_path

        return original_path # 없으면 원래 경로 반환 (이후 로드 실패 처리됨)

    def get_model_structure(self, model_name: str, model_path: str) -> str:
        try:
            module = self._get_model_module(model_name)

            # 경로 해석 (Fallback 적용)
            resolved_path = self._resolve_model_path(model_path)

            model = module.load_model(resolved_path, self.device)
            return str(model)
        except Exception as e:
            return f"Error retrieving model structure: {str(e)}"

    def run_inference(self, model_name: str, model_path: str, image: Image.Image):
        try:
            module = self._get_model_module(model_name)

            # 경로 해석 (Fallback 적용)
            resolved_path = self._resolve_model_path(model_path)

            if os.path.exists(resolved_path):
                print(f"Loading model from: {resolved_path}")
                model = module.load_model(resolved_path, self.device)
            else:
                print(f"Warning: Weight file not found at {resolved_path} (and fallback paths).")
                # 파일이 없어도 초기화된 모델로 진행 (데모용)
                model = module.load_model(resolved_path, self.device)

            result = module.predict(model, image, self.device)
            return result

        except Exception as e:
            print(f"Inference Error: {e}")
            raise e
