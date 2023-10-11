import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# CSV 파일 읽기
df_original = pd.read_csv('data.csv')

# 'measure_date'를 날짜 형식으로 변환
df_original['measure_date'] = pd.to_datetime(df_original['measure_date'])

# 칼럼 선별
selected_columns = ['measure_date', 'algae', 'blue_algae', 'blue-green_algae']
df_col = df_original[selected_columns]

# 2019년 데이터만 필터링
df_col = df_col[df_col['measure_date'].dt.year == 2019]

# # 로우 선별
# df = df_col.iloc[::100]
df = df_col

# 그래프 그리기
plt.figure(figsize=(10, 6))  # 그래프 크기 조절
plt.plot(df['measure_date'], df['algae'], label='algae')
# plt.plot(df['measure_date'], df['blue_algae'], label='blue_algae')
# plt.plot(df['measure_date'], df['blue-green_algae'], label='blue-green_algae')

# 타이틀과 레이블 추가
plt.title('Data from CSV file')
plt.xlabel('Date')
plt.ylabel('Value')
plt.xticks(rotation=45)  # x축 레이블 회전 (날짜가 겹칠 수 있으므로)

# # x축의 레이블을 100개마다 표시
# all_dates = df['measure_date'].tolist()
# selected_dates = all_dates[::10]
# plt.xticks(selected_dates, rotation=45)  # x축 레이블 회전 (날짜가 겹칠 수 있으므로)

# x축의 레이블을 월의 첫 날짜로 설정
monthly_first_dates = pd.date_range(df['measure_date'].min(), df['measure_date'].max(), freq='MS')
plt.xticks(monthly_first_dates, rotation=45)  # x축 레이블 회전 (날짜가 겹칠 수 있으므로)

# 범위 설정
# plt.ylim(0.5, 3)

# 범례 표시
plt.legend()

# 그래프 표시
plt.tight_layout()
plt.grid(True)
plt.show()
