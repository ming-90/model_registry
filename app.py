import streamlit as st
import pandas as pd
from database.sqlite import Database
from src.registry import ModelRegistry
from src.inference import InferenceEngine
import os
from PIL import Image

# 페이지 설정
st.set_page_config(
    page_title="AirsMed Model Registry",
    page_icon="🗄️",
    layout="wide"
)

# 레지스트리 및 인퍼런스 엔진 연결 함수
def get_registry():
    db_path = "database/model_registry.db"
    os.makedirs("database", exist_ok=True)
    db = Database(db_path)
    return ModelRegistry(db)

def get_inference_engine():
    return InferenceEngine()

# 세션 상태 초기화
if 'selected_model_id' not in st.session_state:
    st.session_state.selected_model_id = None
if 'selected_version_id' not in st.session_state:
    st.session_state.selected_version_id = None
if 'upload_model_id' not in st.session_state:
    st.session_state.upload_model_id = None

def view_dashboard(registry):
    st.header("📦 Registered Model Families")

    # 모델 목록 조회
    df_models = registry.get_models_df()

    if df_models.empty:
        st.info("No model families registered. Click 'Initialize Demo Data' in the sidebar.")
        return

    # 모델 검색
    search_query = st.text_input("🔍 Search Model Families", placeholder="e.g. mobilenet")
    if search_query:
        df_models = df_models[df_models['name'].str.contains(search_query, case=False)]

    # 그리드 레이아웃으로 모델 카드 표시
    cols = st.columns(3)
    for idx, row in df_models.iterrows():
        with cols[idx % 3]:
            with st.container(border=True):
                st.subheader(f"📦 {row['name']}")
                st.caption(f"ID: {row['id']}")

                description = row['description'] if row['description'] else "No description available."
                st.write(description)

                st.write(f"**Updated:** {row['updated_at']}")

                if st.button("View Versions", key=f"btn_model_{row['id']}", use_container_width=True):
                    st.session_state.selected_model_id = row['id']
                    st.rerun()

def view_model_versions(registry):
    model_id = st.session_state.selected_model_id

    # 뒤로가기
    if st.button("← Back to Models"):
        st.session_state.selected_model_id = None
        st.rerun()

    df_models = registry.get_models_df()
    model_info = df_models[df_models['id'] == model_id]
    if not model_info.empty:
        model_name = model_info.iloc[0]['name']
        model_desc = model_info.iloc[0]['description']
    else:
        model_name = f"Model ID {model_id}"
        model_desc = ""

    st.title(f"📦 {model_name}")
    if model_desc:
        st.markdown(f"*{model_desc}*")

    st.divider()

    col_ver_header, col_ver_action = st.columns([8, 2])
    with col_ver_header:
        st.subheader("Version History")
    with col_ver_action:
        if st.button("⬆️ New Version", key=f"upload_btn_{model_id}"):
            st.session_state.upload_model_id = model_id
            st.rerun()

    search_version = st.text_input("🔖 Filter by Version", placeholder="e.g. 1.0 or latest")

    df = registry.get_model_versions_df()
    df = df[df['model_id'] == model_id]

    if df.empty:
        st.info("No versions found for this model.")
        return

    if search_version:
        mask = (
            df['version'].str.contains(search_version, case=False, na=False) |
            df['is_latest'].str.contains(search_version, case=False, na=False)
        )
        df = df[mask]

    # 버전 목록을 카드 형태로 표시
    for idx, row in df.iterrows():
        with st.container(border=True):
            col_info, col_btn = st.columns([5, 1])

            with col_info:
                st.markdown(f"### {row['version_display']}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption("Framework")
                    st.write(row['framework'])
                with col2:
                    st.caption("Architecture")
                    st.write(row['architecture'])
                with col3:
                    st.caption("Created At")
                    st.write(row['created_at'])

            with col_btn:
                st.write("")
                st.write("")
                if st.button("View Details", key=f"btn_ver_{row['id']}", use_container_width=True):
                    st.session_state.selected_version_id = row['id']
                    st.rerun()


def view_version_detail(registry, inference_engine):
    version_id = st.session_state.selected_version_id

    if st.button("← Back to Versions"):
        st.session_state.selected_version_id = None
        st.rerun()

    st.title(f"📄 Version Details")

    df = registry.get_model_versions_df()
    model_data = df[df['id'] == version_id]

    if not model_data.empty:
        row = model_data.iloc[0]
        st.header(f"{row['model_name']} v{row['version']}")
        st.markdown(f"**ID:** `{row['id']}`")
        if row['is_latest'] == 'latest':
            st.success("✨ This is the latest version")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Metadata")
            st.write(f"**Framework:** {row['framework']}")
            st.write(f"**Architecture:** {row['architecture']}")

            with st.expander("View Architecture Layers"):
                model_path = os.path.join("artifacts", row['rel_path'])
                structure = inference_engine.get_model_structure(row['model_name'], model_path)
                st.code(structure, language="text")

            st.write(f"**Created At:** {row['created_at']}")
        with col2:
            st.subheader("Input/Output")
            st.write(f"**Input Shape:** `{row['input_shape']}`")
            st.write(f"**Dataset:** {row['dataset']}")

            if row.get('metrics') and row['metrics']:
                try:
                    import json
                    metrics_str = row['metrics'].strip()
                    # 빈 JSON 객체나 빈 문자열이 아닌 경우에만 표시
                    if metrics_str and metrics_str not in ['{}', '']:
                        metrics = json.loads(metrics_str)
                        if metrics:  # 딕셔너리가 비어있지 않은 경우에만 표시
                            st.write("**Metrics:**")
                            for k, v in metrics.items():
                                st.write(f"- {k}: `{v}`")
                except Exception as e:
                    st.warning(f"Failed to parse metrics: {str(e)}")

            if row.get('memo'):
                st.write("**Memo:**")
                st.info(row['memo'])

        st.divider()

        st.subheader("⚡ Model Inference Test")

        test_col1, test_col2 = st.columns([1, 1])

        # 이미지 로딩 로직
        target_image = None
        default_image_path = "assets/test/sample.jpg"

        with test_col1:
            # 탭으로 구분: 업로드 vs 기본 이미지
            tab1, tab2 = st.tabs(["🖼️ Upload Image", "📂 Default Sample"])

            with tab1:
                uploaded_image = st.file_uploader("Upload Test Image", type=["jpg", "png", "jpeg"])
                if uploaded_image:
                    target_image = Image.open(uploaded_image)
                    st.image(target_image, caption="Uploaded Image", use_container_width=True)

            with tab2:
                if os.path.exists(default_image_path):
                    st.info("Using default sample image.")
                    default_image = Image.open(default_image_path)
                    st.image(default_image, caption="Default Sample Image", use_container_width=True)
                    if not uploaded_image:
                        target_image = default_image
                else:
                    st.warning(f"Default sample image not found at `{default_image_path}`.")
                    st.caption("Please add a sample image or use the Upload tab.")

        with test_col2:
            st.write("**Prediction Result**")

            if target_image:
                if st.button("Run Inference", type="primary"):
                    with st.spinner("Running inference..."):
                        try:
                            model_name = row['model_name']
                            model_path = os.path.join("artifacts", row['rel_path'])

                            # Inference Engine을 통해 추론 실행
                            result = inference_engine.run_inference(model_name, model_path, target_image)

                            st.success("Inference Completed!")
                            predicted_class = result['predicted_class']
                            confidence = result['confidence']
                            st.markdown(f"### Predicted Class: `{predicted_class}`")
                            st.progress(confidence, text=f"Confidence: {confidence:.1%}")

                            with st.expander("See Probabilities"):
                                st.json(result['probabilities'])

                        except Exception as e:
                            st.error(f"Inference failed: {str(e)}")
            else:
                st.info("Please upload an image or select the default sample to start inference.")

    else:
        st.error("Model version not found.")

from src.versioning import VersionManager

# ... (중략) ...

def get_version_manager(registry):
    return VersionManager(registry)

# ... (중략) ...

def view_upload_form(registry):
    model_id = st.session_state.upload_model_id
    version_manager = get_version_manager(registry)

    if st.button("← Cancel"):
        st.session_state.upload_model_id = None
        st.rerun()

    st.title("⬆️ Upload New Model Version")

    df = registry.get_models_df()
    target_model = df[df['id'] == model_id]

    if not target_model.empty:
        model_name = target_model.iloc[0]['name']
        st.subheader(f"Target Model: **{model_name}**")
    else:
        model_name = "Unknown"
        st.subheader(f"Target Model ID: {model_id}")

    # 현재 최신 버전 조회
    current_version_str = version_manager.get_latest_version(model_id)
    if current_version_str:
        st.info(f"Current Latest Version: **{current_version_str}**")
    else:
        st.info("No versions found. This will be the first version (1.0.0).")
        current_version_str = "0.0.0"

    with st.form("upload_form"):
        st.write("### Version Strategy")

        version_type = st.radio(
            "Select update type:",
            ["Major", "Minor", "Patch"],
            captions=[
                "Breaking changes / Architecture update",
                "New features / Performance improvement",
                "Bug fixes / Small updates"
            ],
            index=2 # Default to Patch
        )

        # 예상되는 다음 버전 계산
        next_version_str = version_manager.calculate_next_version(current_version_str, version_type.lower())
        st.write(f"**Next Version:** `{next_version_str}`")

        st.write("### Metadata")
        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            framework = st.selectbox("Framework", ["pytorch", "tensorflow", "sklearn", "other"])
            input_shape = st.text_input("Input Shape", value="1x3x224x224")
        with col_meta2:
            # 모델명에 따라 기본값 제안
            default_arch = "UNet" if "unet" in model_name.lower() else "MobileNetV2"
            architecture = st.text_input("Architecture", value=default_arch)
            dataset = st.text_input("Dataset", value="")

        st.write("### Metrics (Optional)")
        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            accuracy = st.number_input("Accuracy", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.4f", help="Enter accuracy between 0 and 1")
        with col_metric2:
            f1_score = st.number_input("F1-Score", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.4f", help="Enter F1-score between 0 and 1")

        st.write("### Memo (Optional)")
        memo = st.text_area("Version Notes", placeholder="Enter any notes about this version...", height=100)

        st.write("### Model File")
        uploaded_file = st.file_uploader("Choose a model file", type=['pt', 'pth', 'pkl', 'h5'])

        submitted = st.form_submit_button("Upload & Register")

        if submitted:
            if uploaded_file is not None:
                # DB Insert Logic
                try:
                    # 1. Parse next version
                    major, minor, patch = map(int, next_version_str.split('.'))

                    # 2. Build metrics JSON from input fields
                    import json
                    import secrets

                    metrics_dict = {}
                    if accuracy > 0:
                        metrics_dict['accuracy'] = accuracy
                    if f1_score > 0:
                        metrics_dict['f1_score'] = f1_score

                    metrics_str = json.dumps(metrics_dict) if metrics_dict else ""

                    # 3. Generate Artifact ID & Path
                    artifact_id = secrets.token_hex(6)
                    rel_path = f"{model_name}/{artifact_id}/model.pth"

                    # 4. Save model file to artifacts directory
                    save_path = os.path.join("artifacts", rel_path)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # 5. DB Insert
                    # Artifact 추가
                    registry.db.insert_artifact(artifact_id, model_name, rel_path)

                    # Version 추가
                    registry.db.insert_model_version(
                        model_id=model_id,
                        version_major=major,
                        version_minor=minor,
                        version_patch=patch,
                        artifact_id=artifact_id,
                        framework=framework,
                        architecture=architecture,
                        input_shape=input_shape,
                        dataset=dataset,
                        metrics=metrics_str.strip(),
                        memo=memo
                    )

                    st.success(f"Successfully registered version {next_version_str}!")
                    st.session_state.upload_model_id = None
                    st.rerun()

                except Exception as e:
                    st.error(f"Registration failed: {e}")
            else:
                st.error("Please upload a file.")

def main():
    st.sidebar.title("Menu")

    if st.sidebar.button("🏠 Home"):
        st.session_state.selected_model_id = None
        st.session_state.selected_version_id = None
        st.session_state.upload_model_id = None
        st.rerun()

    registry = get_registry()
    inference_engine = get_inference_engine()

    if st.sidebar.button("Initialize Demo Data"):
        with st.spinner("Inserting demo data..."):
            registry.insert_demo_data()
        st.sidebar.success("Demo data initialized!")
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()

    if st.session_state.upload_model_id is not None:
        view_upload_form(registry)

    elif st.session_state.selected_version_id is not None:
        view_version_detail(registry, inference_engine)

    elif st.session_state.selected_model_id is not None:
        view_model_versions(registry)

    else:
        view_dashboard(registry)

if __name__ == "__main__":
    main()
