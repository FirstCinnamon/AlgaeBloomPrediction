import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import matplotlib.style as style
matplotlib.use('TkAgg')

show_graph = False
show_year_graph = False
show_tree = False
show_in_year = True

# 스타일 선언
style.use('ggplot')

# CSV 파일 읽기
data = pd.read_csv('data.csv')

# up_T-N 을 up_T-P 로 나누어서 'N/P' 칼럼에 저장
data['N/P'] = data['up_T-N'] / data['up_T-P']

value = 'diatomeae'

# 칼럼 선별
selected_columns = ['measure_date', value]
df_selected = data[selected_columns]

# 날짜로 변환
df_selected.loc[:, 'measure_date'] = pd.to_datetime(df_selected['measure_date'])

# 날짜 선별
start_date = '2019-01-01'
end_date = '2021-12-31'
df_selected = df_selected[(df_selected['measure_date'] >= start_date) & (df_selected['measure_date'] <= end_date)]


def show_year_graph(value):
    # 연도별로 데이터를 분리
    years = [2019, 2020, 2021]
    for year in years:
        df_year = df_selected[df_selected['measure_date'].dt.year == year]

        # 그래프 그리기
        plt.figure(figsize=(10, 6))  # 그래프 크기 조절
        plt.plot(df_year['measure_date'], df_year[value], label=value)

        # 타이틀과 레이블 추가
        plt.title(f'Data for {year}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.xticks(rotation=45)  # x축 레이블 회전 (날짜가 겹칠 수 있으므로)

        # 범례 표시
        plt.legend()

        # 그래프 저장
        plt.savefig(f"data_{year}.png")

        # 그래프 표시
        plt.tight_layout()
        plt.grid(True)
        plt.show()

# show_year_graph(value)

def show_in_year(value):
    plt.figure(figsize=(12, 7))

    # 2019년, 2020년, 2022년의 데이터 선택
    df_2019 = df_selected[df_selected['measure_date'].dt.year == 2019]
    df_2020 = df_selected[df_selected['measure_date'].dt.year == 2020]
    df_2021 = df_selected[df_selected['measure_date'].dt.year == 2021]

    # 'measure_date'를 월-일 형식으로 변환
    x_dates_2019 = df_2019['measure_date'].dt.strftime('%m-%d').tolist()
    x_dates_2020 = df_2020['measure_date'].dt.strftime('%m-%d').tolist()
    x_dates_2021 = df_2021['measure_date'].dt.strftime('%m-%d').tolist()

    # 2019년, 2020년, 2022년 모두에 존재하는 날짜를 고려
    common_x_dates = list(set(x_dates_2019 + x_dates_2020 + x_dates_2021))
    common_x_dates.sort()

    # 그래프 그리기
    y_values_2019 = [df_2019[df_2019['measure_date'].dt.strftime('%m-%d') == date][value].values[
                         0] if date in x_dates_2019 else None for date in common_x_dates]
    y_values_2020 = [df_2020[df_2020['measure_date'].dt.strftime('%m-%d') == date][value].values[
                         0] if date in x_dates_2020 else None for date in common_x_dates]
    y_values_2021 = [df_2021[df_2021['measure_date'].dt.strftime('%m-%d') == date][value].values[
                         0] if date in x_dates_2021 else None for date in common_x_dates]

    plt.plot(common_x_dates, y_values_2019, color='red', label=value+"2019")
    plt.plot(common_x_dates, y_values_2020, color='blue', label=value+"2020")
    plt.plot(common_x_dates, y_values_2021, color='green', label=value+"2021")  # 2021년 데이터를 초록색으로 추가

    # 월의 첫 날짜만 x축 눈금으로 선택하기 위한 리스트 생성
    first_days_of_month = [f"0{i}-01" if i < 10 else f"{i}-01" for i in range(1, 13)]

    # 타이틀, 레이블, 범례 설정
    plt.title('Data for 2019, 2020, and 2021')
    plt.xlabel('Month-Day')
    plt.ylabel('Value')

    # x축 레이블 회전 및 월의 첫 날짜만 눈금으로 표시
    plt.xticks(first_days_of_month, rotation=45)

    plt.legend()

    # 그래프 저장 및 출력
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("data_2019_2020_2021.png")
    plt.show()

def show_average(value):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(12, 7))

    # 2019년, 2020년, 2021년의 데이터 선택
    df_2019 = df_selected[df_selected['measure_date'].dt.year == 2019]
    df_2020 = df_selected[df_selected['measure_date'].dt.year == 2020]
    df_2021 = df_selected[df_selected['measure_date'].dt.year == 2021]

    # 'measure_date'를 월-일 형식으로 변환
    x_dates_2019 = df_2019['measure_date'].dt.strftime('%m-%d').tolist()
    x_dates_2020 = df_2020['measure_date'].dt.strftime('%m-%d').tolist()
    x_dates_2021 = df_2021['measure_date'].dt.strftime('%m-%d').tolist()

    # 2019년, 2020년, 2021년 모두에 존재하는 날짜를 고려
    common_x_dates = list(set(x_dates_2019 + x_dates_2020 + x_dates_2021))
    common_x_dates.sort()

    # 평균값 계산
    avg_values = []
    for date in common_x_dates:
        vals = []
        if date in x_dates_2019:
            vals.append(df_2019[df_2019['measure_date'].dt.strftime('%m-%d') == date][value].values[0])
        if date in x_dates_2020:
            vals.append(df_2020[df_2020['measure_date'].dt.strftime('%m-%d') == date][value].values[0])
        if date in x_dates_2021:
            vals.append(df_2021[df_2021['measure_date'].dt.strftime('%m-%d') == date][value].values[0])
        avg_values.append(np.mean(vals))

    # 평균값을 그래프로 그리기
    plt.plot(common_x_dates, avg_values, color='purple', label='Average'+value+'2019-2021')

    # 월의 첫 날짜만 x축 눈금으로 선택하기 위한 리스트 생성
    first_days_of_month = [f"0{i}-01" if i < 10 else f"{i}-01" for i in range(1, 13)]

    # 타이틀, 레이블, 범례 설정
    plt.title('Average '+value+' for 2019, 2020, and 2021')
    plt.xlabel('Month-Day')
    plt.ylabel('Value')

    # x축 레이블 회전 및 월의 첫 날짜만 눈금으로 표시
    plt.xticks(first_days_of_month, rotation=45)

    plt.legend()

    # 그래프 저장 및 출력
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("average_data_2019_2020_2021.png")
    plt.show()


def show_averages(values):
    import matplotlib.pyplot as plt
    import numpy as np

    if len(values) != 2:
        raise ValueError("This function currently supports only 2 values for dual y-axes.")

    # 칼럼 선별
    selected_columns = ['measure_date'] + values
    df_selected = data[selected_columns]
    df_selected['measure_date'] = pd.to_datetime(df_selected['measure_date'])

    plt.figure(figsize=(12, 7))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    for idx, value in enumerate(values):
        # 데이터 선택 및 연도별 분리
        df_2019 = df_selected[df_selected['measure_date'].dt.year == 2019]
        df_2020 = df_selected[df_selected['measure_date'].dt.year == 2020]
        df_2021 = df_selected[df_selected['measure_date'].dt.year == 2021]

        # 연도별 고유한 날짜 추출
        common_x_dates = sorted(set(df_2019['measure_date'].dt.strftime('%m-%d').tolist() +
                                    df_2020['measure_date'].dt.strftime('%m-%d').tolist() +
                                    df_2021['measure_date'].dt.strftime('%m-%d').tolist()))

        # 연도별 평균값 계산
        avg_values = []
        for date in common_x_dates:
            vals = []
            if date in df_2019['measure_date'].dt.strftime('%m-%d').tolist():
                vals.append(df_2019[df_2019['measure_date'].dt.strftime('%m-%d') == date][value].values[0])
            if date in df_2020['measure_date'].dt.strftime('%m-%d').tolist():
                vals.append(df_2020[df_2020['measure_date'].dt.strftime('%m-%d') == date][value].values[0])
            if date in df_2021['measure_date'].dt.strftime('%m-%d').tolist():
                vals.append(df_2021[df_2021['measure_date'].dt.strftime('%m-%d') == date][value].values[0])
            avg_values.append(np.mean(vals))

        if idx == 0:
            ax1.plot(common_x_dates, avg_values, color='purple', label=f'Average {value} 2019-2021')
            ax1.set_ylabel(value, color='purple')
            ax1.tick_params(axis='y', labelcolor='purple')
        else:
            ax2.plot(common_x_dates, avg_values, color='green', label=f'Average {value} 2019-2021')
            ax2.set_ylabel(value, color='green')
            ax2.tick_params(axis='y', labelcolor='green')

    # 월의 첫 날짜만 x축 눈금으로 선택하기 위한 리스트 생성
    first_days_of_month = [f"0{i}-01" if i < 10 else f"{i}-01" for i in range(1, 13)]
    plt.xticks(first_days_of_month, rotation=45)

    # 타이틀, 레이블 설정
    plt.title('Average Values for 2019, 2020, and 2021')
    plt.xlabel('Month-Day')

    # 범례 설정 및 그래프 출력
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    plt.grid(True)
    plt.savefig("average_data_2019_2020_2021_dual_yaxes.png")
    plt.show()


#
def show_graph(value):
    plt.figure(figsize=(12, 7))

    # 2019년과 2020년의 데이터 선택
    df_2019 = df_selected[df_selected['measure_date'].dt.year == 2019]
    df_2020 = df_selected[df_selected['measure_date'].dt.year == 2020]

    # 'measure_date'를 월-일 형식으로 변환
    x_dates_2019 = df_2019['measure_date'].dt.strftime('%m-%d').tolist()
    x_dates_2020 = df_2020['measure_date'].dt.strftime('%m-%d').tolist()

    # 공통 x축 (월-일) 정의
    common_x_dates = [f"{str(month).zfill(2)}-{str(day).zfill(2)}" for month in range(1, 13) for day in range(1, 32)]

    # 변환된 날짜를 X축으로 사용하여 빨간색(2019년)과 파란색(2020년)으로 그래프를 그립니다.
    plt.plot(common_x_dates, [df_2019[df_2019['measure_date'].dt.strftime('%m-%d') == date]['up_T-P'].values[
                                  0] if date in x_dates_2019 else None for date in common_x_dates], color='red',
             label='up_T-P 2019')
    plt.plot(common_x_dates, [df_2020[df_2020['measure_date'].dt.strftime('%m-%d') == date]['up_T-P'].values[
                                  0] if date in x_dates_2020 else None for date in common_x_dates], color='blue',
             label='up_T-P 2020')

    # 월의 첫 날짜만 x축 눈금으로 선택하기 위한 리스트 생성
    first_days_of_month = [f"0{i}-01" if i < 10 else f"{i}-01" for i in range(1, 13)]

    # 타이틀, 레이블, 범례 설정
    plt.title('Data for 2019 and 2020')
    plt.xlabel('Month-Day')
    plt.ylabel('Value')

    # x축 레이블 회전 및 월의 첫 날짜만 눈금으로 표시
    plt.xticks(first_days_of_month, rotation=45)

    plt.legend()

    # 그래프 저장 및 출력
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("data_2019_2020.png")
    plt.show()

def show_monthly_average_graphs(values):
    # 대상 연도 리스트 정의
    target_years = [2019, 2020, 2021]

    plt.figure(figsize=(12, 7))

    months = list(range(1, 13))
    month_labels = [f"{month}월" for month in months]

    for year in target_years:
        df_year = data[data['measure_date'].dt.year == year]

        # 월별 평균값을 계산
        monthly_avg = df_year.groupby(df_year['measure_date'].dt.month)[values].mean()

        # 각 값에 대해 선 그래프 그리기
        for value in values:
            plt.plot(monthly_avg.index, monthly_avg[value], label=f"{value} ({year}년)")

    # 타이틀, 레이블, 범례 설정
    plt.title('월별 평균값: 2019, 2020, 2021년')
    plt.xlabel('월')
    plt.ylabel('평균값')
    plt.xticks(months, month_labels)  # X 축의 눈금 레이블 설정
    plt.legend()

    # 그래프 저장 및 출력
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("monthly_average_data_2019_2020_2021.png")
    plt.show()

#
# '''
# ---------------------------------
# '''
#
def show_tree():
    DEPTH = 8
    normalize = True

    # 특정 칼럼을 시계열 데이터로 처리
    time_column = 'measure_date'
    df_selected.loc[:, time_column] = pd.to_datetime(data[time_column])

    # 타겟 변수 지정
    target_column = 'algae'
    X = df_selected.drop(columns=[target_column, time_column]).values  # 타겟과 시계열 칼럼 제외
    y = df_selected[target_column].values

    # 학습 데이터와 테스트 데이터로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
                                                        shuffle=False)  # shuffle=False는 시계열 데이터를 고려

    if normalize:
        # 스케일러 초기화 및 적용
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # 결정 트리 모델 생성 및 학습
    tree_clf = DecisionTreeRegressor(max_depth=DEPTH)
    tree_clf.fit(X_train, y_train)

    # 예측
    y_pred = tree_clf.predict(X_test)

    # RMSE 계산
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse:.4f}")

    # 예측
    y_pred = tree_clf.predict(X_test)

    # RMSE 계산
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse:.4f}")



    # 결정 트리 시각화
    plt.figure(figsize=(12, 8))
    plot_tree(tree_clf, filled=True, fontsize=10)
    plt.show()

    # 예측 값과 실제 값을 그래프로 표시
    plt.figure(figsize=(10, 6))
    plt.plot(df_selected['measure_date'][-len(y_test):], y_test, label='Actual Values', color='blue')
    plt.plot(df_selected['measure_date'][-len(y_test):], y_pred, label='Predicted Values', color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    plt.show()

# show_tree()

# show_in_year(value)
# show_average(value)
# show_monthly_average_graphs(['blue_algae', 'blue-green_algae', 'diatomeae'])

# 함수 호출 예제:
show_averages(['algae', 'diatomeae'])

