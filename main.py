import streamlit as st
import pandas as pd


def main():
    st.title("녹조 예측 모델 파라미터 설정")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # 데이터 로드
        data = pd.read_csv(uploaded_file)
        # 데이터 프리뷰 표시
        st.write(data.head())

        # 사용자가 칼럼을 선택할 수 있는 멀티셀렉트 위젯 추가
        selected_columns = st.multiselect("Choose columns", data.columns.tolist(), default=data.columns.tolist())

        # 선택된 칼럼으로 구성된 데이터프레임 보여주기
        st.dataframe(data[selected_columns])

        '''
        -------------------- 이하부터 하이퍼파라미터 설정 ----------------------
        '''

        st.subheader("Set Hyperparameters for Transformer Model")

        # num_layers: 트랜스포머 레이어 수 설정
        # 각 레이어에 멀티-헤드 어텐션 메커니즘과 포지션 와이즈 피드 포워드 네트워크가 있음
        num_layers = st.slider("Number of Transformer Layers", min_value=1, max_value=6, value=3, step=1)

        # d_model: 입력 시퀀스 원소와 내부 레이어 출력 차원 설정
        # 어텐션 메커니즘과 피드포워드 네트워크의 차원 정의
        d_model = st.slider("Dimension of Model", min_value=128, max_value=512, value=256, step=128)

        # num_heads: 멀티헤드 어텐션 병렬 헤드 수 설정
        # 각 헤드가 독립적으로 어텐션 계산하고 결과를 d_model 차원으로 연결
        num_heads = st.slider("Number of Attention Heads", min_value=1, max_value=8, value=4, step=1)

        # dff: 포지션 와이즈 피드포워드 네트워크 내의 두 개 완전연결층 은닉층 차원 설정
        # 첫 번째 완전연결층은 차원을 dff로 확장하고, 두 번째 완전연결층은 모델 차원으로 다시 축소
        dff = st.slider("Dimension of Feedforward Network", min_value=512, max_value=2048, value=1024, step=512)

        # dropout_rate: 모델 각 층에서 적용할 드롭아웃 확률 설정
        # 드롭아웃은 과적합 방지에 도움
        dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.1, step=0.1)


        # 선택된 하이퍼파라미터 확인
        st.markdown(
            f"**Confirmed Hyperparameters**\n- Num Layers: {num_layers}\n- Model Dimension: {d_model}\n- Num Attention Heads: {num_heads}\n- Feedforward Network Dimension: {dff}\n- Dropout Rate: {dropout_rate}")

        '''
        -------------------- 이상 하이퍼파라미터 설정 끝 ----------------------
        '''

        # 추가로 모델 학습 및 평가 버튼을 생성하여, 이를 클릭할 경우 실제 학습 및 평가
        if st.button("Train & Evaluate Model"):
            st.write("Training and evaluating the model with chosen parameters...")  # 실제 학습 및 평가 코드를 여기에 구현
            # 예시로, 함수형태로 학습 및 평가 코드를 구현하고, 버튼 클릭 시 실행
            # train_and_evaluate_model(data[selected_columns], num_layers, d_model, num_heads, dff, dropout_rate)


if __name__ == "__main__":
    main()
