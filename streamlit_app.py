import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# 페이지 설정
st.set_page_config(
    page_title="감정 인식 앱",
    page_icon="😊",
    layout="wide"
)

# 감정별 이모지 및 피드백 메시지
EMOTION_FEEDBACK = {
    'happy': {
        'emoji': '😊',
        'name': '행복',
        'message': '행복해 보이시네요! 긍정적인 에너지가 느껴집니다.',
        'color': '#FFD700'
    },
    'sad': {
        'emoji': '😢',
        'name': '슬픔',
        'message': '슬퍼 보이시네요. 힘내세요! 좋은 일이 생길 거예요.',
        'color': '#4169E1'
    },
    'angry': {
        'emoji': '😠',
        'name': '화남',
        'message': '화가 나 보이시네요. 심호흡을 하고 진정하세요.',
        'color': '#DC143C'
    },
    'surprise': {
        'emoji': '😲',
        'name': '놀람',
        'message': '놀라신 것 같네요! 무슨 일이 있으셨나요?',
        'color': '#FF8C00'
    },
    'fear': {
        'emoji': '😨',
        'name': '두려움',
        'message': '두려워 보이시네요. 괜찮으실 거예요.',
        'color': '#9370DB'
    },
    'disgust': {
        'emoji': '🤢',
        'name': '혐오',
        'message': '불쾌해 보이시네요. 기분 전환이 필요할 것 같아요.',
        'color': '#228B22'
    },
    'neutral': {
        'emoji': '😐',
        'name': '무표정',
        'message': '평온한 상태시네요. 안정적인 감정 상태입니다.',
        'color': '#808080'
    }
}

def analyze_emotion(image):
    """이미지에서 감정 분석"""
    try:
        # PIL Image를 numpy array로 변환
        img_array = np.array(image)
        
        # RGB to BGR (OpenCV format)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            cv2.imwrite(tmp_file.name, img_array)
            tmp_path = tmp_file.name
        
        try:
            # DeepFace로 감정 분석
            result = DeepFace.analyze(
                img_path=tmp_path,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            # 결과가 리스트인 경우 첫 번째 요소 사용
            if isinstance(result, list):
                result = result[0]
            
            return result
        finally:
            # 임시 파일 삭제
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        st.error(f"감정 분석 중 오류가 발생했습니다: {str(e)}")
        return None

def display_emotion_result(result):
    """감정 분석 결과 표시"""
    if result is None:
        return
    
    emotions = result.get('emotion', {})
    dominant_emotion = result.get('dominant_emotion', 'neutral')
    
    # 감정별 피드백 정보 가져오기
    feedback = EMOTION_FEEDBACK.get(dominant_emotion, EMOTION_FEEDBACK['neutral'])
    
    # 주요 감정 표시
    st.markdown(f"### {feedback['emoji']} 주요 감정: **{feedback['name']}**")
    st.markdown(f"<p style='color: {feedback['color']}; font-size: 18px;'>{feedback['message']}</p>", 
                unsafe_allow_html=True)
    
    # 모든 감정 확률 표시
    st.markdown("---")
    st.subheader("📊 감정 분석 결과")
    
    # 감정을 확률 순으로 정렬
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    
    for emotion, score in sorted_emotions:
        col1, col2 = st.columns([1, 4])
        with col1:
            emotion_info = EMOTION_FEEDBACK.get(emotion, {})
            emoji = emotion_info.get('emoji', '😶')
            name = emotion_info.get('name', emotion)
            st.write(f"{emoji} {name}")
        with col2:
            st.progress(float(score) / 100)
            st.caption(f"{score:.2f}%")

# 앱 제목
st.title("😊 감정 인식 앱")
st.markdown("얼굴 사진을 통해 감정을 분석하고 피드백을 제공합니다.")

# 탭 생성
tab1, tab2 = st.tabs(["📷 카메라로 촬영", "📁 이미지 업로드"])

# 카메라 탭
with tab1:
    st.subheader("카메라로 얼굴 사진 촬영")
    st.info("카메라를 활성화하고 촬영 버튼을 클릭하세요.")
    
    # 세션 상태 초기화
    if 'camera_captured' not in st.session_state:
        st.session_state.camera_captured = False
    if 'show_camera' not in st.session_state:
        st.session_state.show_camera = False
    
    # 카메라 활성화 버튼
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        if st.button("📷 카메라 켜기", use_container_width=True):
            st.session_state.show_camera = True
            st.session_state.camera_captured = False
    with col_btn2:
        if st.button("❌ 카메라 끄기", use_container_width=True):
            st.session_state.show_camera = False
            st.session_state.camera_captured = False
    
    # 카메라 입력
    if st.session_state.show_camera:
        camera_photo = st.camera_input("사진 촬영")
        
        if camera_photo is not None:
            # 이미지 열기
            image = Image.open(camera_photo)
            
            # 감정 분석 버튼
            if st.button("🔍 감정 분석하기", type="primary", use_container_width=True):
                st.session_state.camera_captured = True
            
            if st.session_state.camera_captured:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption="촬영된 이미지", use_container_width=True)
                
                with col2:
                    with st.spinner("감정 분석 중..."):
                        result = analyze_emotion(image)
                        if result:
                            display_emotion_result(result)

# 이미지 업로드 탭
with tab2:
    st.subheader("이미지 파일 업로드")
    st.info("JPG, JPEG, PNG 형식의 얼굴 사진을 업로드하세요.")
    
    uploaded_file = st.file_uploader(
        "이미지 선택",
        type=['jpg', 'jpeg', 'png'],
        help="얼굴이 포함된 이미지를 업로드하세요."
    )
    
    if uploaded_file is not None:
        # 이미지 열기
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="업로드된 이미지", use_container_width=True)
        
        with col2:
            with st.spinner("감정 분석 중..."):
                result = analyze_emotion(image)
                if result:
                    display_emotion_result(result)

# 사이드바 정보
with st.sidebar:
    st.header("ℹ️ 사용 방법")
    st.markdown("""
    1. **카메라로 촬영** 또는 **이미지 업로드** 탭을 선택하세요.
    2. 얼굴이 잘 보이도록 사진을 촬영하거나 업로드하세요.
    3. 자동으로 감정이 분석되고 결과가 표시됩니다.
    
    ---
    
    ### 인식 가능한 감정
    - 😊 행복 (Happy)
    - 😢 슬픔 (Sad)
    - 😠 화남 (Angry)
    - 😲 놀람 (Surprise)
    - 😨 두려움 (Fear)
    - 🤢 혐오 (Disgust)
    - 😐 무표정 (Neutral)
    
    ---
    
    ### 📝 팁
    - 얼굴이 정면을 향하도록 촬영하세요
    - 조명이 적절한 환경에서 촬영하세요
    - 얼굴이 가려지지 않도록 하세요
    """)
    
    st.markdown("---")
    st.caption("Powered by DeepFace & Streamlit")

