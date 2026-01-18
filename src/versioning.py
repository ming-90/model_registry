class VersionManager:
    def __init__(self, registry):
        self.registry = registry

    def get_latest_version(self, model_id):
        """특정 모델의 최신 버전을 문자열로 반환 (예: '1.2.0')"""
        df = self.registry.get_model_versions_df()
        if df.empty:
            return None

        model_versions = df[df['model_id'] == model_id]
        if model_versions.empty:
            return None

        # is_latest가 'latest'인 행 찾기
        latest_row = model_versions[model_versions['is_latest'] == 'latest']
        if not latest_row.empty:
            return latest_row.iloc[0]['version']

        # 만약 'latest' 마킹이 없다면 정렬해서 첫 번째 가져오기 (이미 registry에서 정렬됨)
        return model_versions.iloc[0]['version']

    def calculate_next_version(self, current_version_str, update_type):
        """현재 버전과 업데이트 타입(major, minor, patch)에 따라 다음 버전을 계산"""
        if not current_version_str:
            return "1.0.0"

        try:
            major, minor, patch = map(int, current_version_str.split('.'))
        except ValueError:
            return "1.0.0"

        update_type = update_type.lower()

        if update_type == 'major':
            return f"{major + 1}.0.0"
        elif update_type == 'minor':
            return f"{major}.{minor + 1}.0"
        elif update_type == 'patch':
            return f"{major}.{minor}.{patch + 1}"
        else:
            return current_version_str

