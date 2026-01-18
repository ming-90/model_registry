import pandas as pd
from database.sqlite import Database

class ModelRegistry:
    def __init__(self, db: Database):
        """Model Registry 초기화"""
        self.db = db

    def get_models_df(self):
        """모델 목록 DataFrame 반환"""
        rows, columns = self.db.get_models()
        if not rows:
            return pd.DataFrame(columns=columns if columns else [])

        df = pd.DataFrame(rows, columns=columns)
        return df

    def get_model_versions_df(self):
        """
        모든 모델 버전 정보를 가공된 DataFrame으로 반환
        - DB 조회
        - DataFrame 변환
        - 'version_display' 컬럼 생성 (latest 태그 포함)
        - 결측치 처리
        """
        rows, columns = self.db.get_all_model_versions()

        if not rows:
            return pd.DataFrame(columns=columns if columns else [])

        df = pd.DataFrame(rows, columns=columns)

        # 후처리 1: version 컬럼 포맷팅 (latest 태그 추가)
        if 'is_latest' in df.columns and 'version' in df.columns:
            df['version_display'] = df.apply(
                lambda x: f"{x['version']} ({x['is_latest']})" if x['is_latest'] == 'latest' else x['version'],
                axis=1
            )
        else:
            df['version_display'] = df['version']

        # 후처리 2: 결측치 처리
        if 'model_name' in df.columns:
            df['model_name'] = df['model_name'].fillna('Unknown Model')
        if 'model_description' in df.columns:
            df['model_description'] = df['model_description'].fillna('')

        return df

    def insert_model(
        self, name, description,
        version_major, version_minor, version_patch,
        framework, architecture, input_shape, dataset, file):
        """모델 추가"""

        # 1. 파일 저장
        artifact_id = secrets.token_hex(6)
        rel_path = f"{name}/{artifact_id}"
        save_path = os.path.join("artifacts", rel_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 2. Model 추가
        model_id = self.db.insert_model(name, description)
        if model_id is None:
            raise Exception("Model insertion failed")

        # 3. Artifact 추가
        artifact_id = self.db.insert_artifact(artifact_id, model_name, rel_path)
        if artifact_id is None:
            raise Exception("Artifact insertion failed")

        # 4. Model Version 추가
        model_version_id = self.db.insert_model_version(
            model_id, version_major, version_minor, version_patch,
            artifact_id, framework, architecture, input_shape, dataset)
        if model_version_id is None:
            raise Exception("Model version insertion failed")


    def insert_demo_data(self):
        """데모 데이터 추가 (Database 클래스에 위임)"""
        self.db.insert_demo_data()
