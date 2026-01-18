import sqlite3
from sqlite3 import Error

class Database:
    def __init__(self, db_file):
        """데이터베이스 연결 초기화"""
        self.db_file = db_file
        self.conn = None
        try:
            self.conn = sqlite3.connect(db_file, check_same_thread=False) # Streamlit에서 스레드 문제 방지
            # 외래 키 제약 조건 활성화
            self.conn.execute("PRAGMA foreign_keys = ON;")
            print(f"SQLite 데이터베이스 연결 성공: {db_file}")
            self.create_tables()
        except Error as e:
            print(f"데이터베이스 연결 오류: {e}")

    def create_tables(self):
        """테이블 생성"""

        # create models table
        query = """
            CREATE TABLE IF NOT EXISTS models (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL UNIQUE,      -- 예: "mobilenet", "unet"
                base_model  TEXT NOT NULL,            -- 예: "mobilenet_v2", "UNet"
                description TEXT,
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """
        self.conn.execute(query)

        # create store table
        query = """
            CREATE TABLE IF NOT EXISTS artifacts (
                id               TEXT PRIMARY KEY,         -- UUID 문자열
                model_name       TEXT NOT NULL,            -- 편의상 중복 보관 (store/mobilenet/{uuid}/...)
                rel_path         TEXT NOT NULL,            -- 예: "mobilenet/{uuid}/model.pth"
                created_at       TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """
        self.conn.execute(query)

        # create model_versions table
        query = """
            CREATE TABLE IF NOT EXISTS model_versions (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id       INTEGER NOT NULL,          -- FK -> models.id
                version_major  INTEGER NOT NULL,
                version_minor  INTEGER NOT NULL,
                version_patch  INTEGER NOT NULL,
                artifact_id    TEXT NOT NULL,             -- FK -> artifacts.id (UUID)

                framework      TEXT NOT NULL,             -- 예: "pytorch"
                architecture   TEXT,                      -- 예: "torchvision.mobilenet_v2", "UNetSmall"
                input_shape    TEXT,                      -- 예: "1x3x224x224" (또는 JSON 문자열)
                dataset        TEXT,                      -- 예: "MedMNIST/pathmnist v1"
                metrics        TEXT,                      -- JSON 문자열 예: {"accuracy": 0.95, "loss": 0.1}
                memo           TEXT,                      -- 버전 메모/노트
                created_at     TEXT NOT NULL DEFAULT (datetime('now')),

                FOREIGN KEY (model_id)  REFERENCES models(id),
                FOREIGN KEY (artifact_id) REFERENCES artifacts(id),
                CONSTRAINT uq_model_version UNIQUE (model_id, version_major, version_minor, version_patch)
            );
        """
        self.conn.execute(query)
        self.conn.commit()

    def insert_demo_data(self):
        """데이터 추가 (Create)"""

        # 이미 데이터가 있는지 확인
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT count(*) FROM model_versions")
            if cur.fetchone()[0] > 0:
                print("Demo data already exists.")
                return
        except Error as e:
            print(f"데이터 확인 오류: {e}")
            return

        # 1. Artifacts 데이터 추가
        # 경로 구조: {model_name}/{uuid}/model.pth
        try:
            artifacts_data = [
                ('967eea071b43', 'mobilenet', 'mobilenet/967eea071b43/model.pth'),
                ('fad74e49128e', 'mobilenet', 'mobilenet/fad74e49128e/model.pth'),
                ('46909a0f58b7', 'unet', 'unet/46909a0f58b7/model.pth')
            ]
            self.conn.executemany(
                "INSERT OR IGNORE INTO artifacts (id, model_name, rel_path) VALUES (?, ?, ?)",
                artifacts_data
            )
            self.conn.commit()
            print("Artifacts data inserted.")
        except Error as e:
            print(f"Artifacts 데이터 추가 오류: {e}")

        # 2. Models 데이터 추가
        try:
            models_data = [
                (1, 'mobilenet', 'mobilenet_v2', 'MobileNet V2 model for general purpose vision tasks'),
                (2, 'unet', 'unet', 'U-Net for medical image segmentation')
            ]
            self.conn.executemany(
                "INSERT OR IGNORE INTO models (id, name, base_model, description) VALUES (?, ?, ?, ?)",
                models_data
            )
            self.conn.commit()
            print("Models data inserted.")
        except Error as e:
            print(f"모델 데이터 추가 오류: {e}")

        # 3. Model Versions 데이터 추가
        query = """
            INSERT INTO model_versions (
                model_id,
                version_major,
                version_minor,
                version_patch,
                artifact_id,
                framework,
                architecture,
                input_shape,
                dataset,
                metrics,
                memo
            )
            VALUES
            (1, 1, 0, 0, '967eea071b43', 'pytorch', 'MobileNetV2', '1x3x224x224', 'ImageNet-1k', '{"accuracy": 0.71, "top5_accuracy": 0.90}', 'Initial release with ImageNet weights'),
            (1, 1, 1, 0, 'fad74e49128e', 'pytorch', 'MobileNetV2', '1x3x224x224', 'CustomDataset v1', '{"accuracy": 0.85, "f1_score": 0.84}', 'Fine-tuned on custom medical dataset'),
            (2, 1, 0, 0, '46909a0f58b7', 'pytorch', 'UNet', '1x1x64x64', 'Medical/Dataset v1', '{"dice_score": 0.89, "iou": 0.82}', 'Standard UNet implementation for segmentation');
        """
        try:
            self.conn.execute(query)
            self.conn.commit()
            print("Demo data inserted.")
        except Error as e:
            print(f"버전 데이터 추가 오류: {e}")

    def get_models(self):
        """모델 목록 조회"""
        query = """
            SELECT id, name, description, created_at, updated_at
            FROM models
            ORDER BY name ASC
        """
        try:
            cur = self.conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
            if cur.description:
                column_names = [description[0] for description in cur.description]
            else:
                column_names = []
            return rows, column_names
        except Error as e:
            print(f"모델 목록 조회 오류: {e}")
            return [], []

    def get_all_model_versions(self):
        """모든 모델 버전 정보 조회"""
        # artifacts 테이블 조인하여 rel_path 가져오기
        query = """
            SELECT
                mv.id,
                mv.model_id,
                m.name as model_name,
                m.description as model_description,
                mv.version_major || '.' || mv.version_minor || '.' || mv.version_patch as version,
                mv.framework,
                mv.architecture,
                mv.input_shape,
                mv.dataset,
                mv.metrics,
                mv.memo,
                mv.created_at,
                mv.artifact_id,
                mv.artifact_id,
                a.rel_path,
                CASE
                    WHEN ROW_NUMBER() OVER (
                        PARTITION BY mv.model_id
                        ORDER BY mv.version_major DESC, mv.version_minor DESC, mv.version_patch DESC
                    ) = 1 THEN 'latest'
                    ELSE ''
                END as is_latest
            FROM model_versions mv
            LEFT JOIN models m ON mv.model_id = m.id
            LEFT JOIN artifacts a ON mv.artifact_id = a.id
            ORDER BY mv.model_id ASC, mv.version_major DESC, mv.version_minor DESC, mv.version_patch DESC
        """
        try:
            cur = self.conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
            if cur.description:
                column_names = [description[0] for description in cur.description]
            else:
                column_names = []
            return rows, column_names
        except Error as e:
            print(f"조회 오류: {e}")
            return [], []

    def insert_model(self, name: str, description: str) -> int:
        """모델 메타데이터 추가 (models 테이블)"""
        query = """
            INSERT INTO models (name, description)
            VALUES (?, ?)
        """
        try:
            cur = self.conn.cursor()
            cur.execute(query, (name, description))
            self.conn.commit()
            return cur.lastrowid
        except Error as e:
            print(f"모델 추가 오류: {e}")
            return None

    def insert_artifact(self, artifact_id: str, model_name: str, rel_path: str):
        """아티팩트 정보 추가 (artifacts 테이블)"""
        query = """
            INSERT INTO artifacts (id, model_name, rel_path)
            VALUES (?, ?, ?)
        """
        try:
            self.conn.execute(query, (artifact_id, model_name, rel_path))
            self.conn.commit()
        except Error as e:
            print(f"아티팩트 추가 오류: {e}")
            raise e

    def insert_model_version(self, model_id: int, version_major: int, version_minor: int, version_patch: int,
                             artifact_id: str, framework: str, architecture: str, input_shape: str = "", dataset: str = "", metrics: str = "", memo: str = ""):
        """모델 버전 정보 추가 (model_versions 테이블)"""
        query = """
            INSERT INTO model_versions (
                model_id, version_major, version_minor, version_patch,
                artifact_id, framework, architecture, input_shape, dataset, metrics, memo
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            self.conn.execute(query, (
                model_id, version_major, version_minor, version_patch,
                artifact_id, framework, architecture, input_shape, dataset, metrics, memo
            ))
            self.conn.commit()
        except Error as e:
            print(f"버전 추가 오류: {e}")
            raise e

    def close(self):
        """데이터베이스 연결 종료"""
        if self.conn:
            self.conn.close()
            print("SQLite 데이터베이스 연결 종료")
