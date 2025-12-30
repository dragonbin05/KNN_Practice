import pandas as pd
import streamlit as st
import pyupbit
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

def load_btc_data(today):
    """
    Docstring for load_btc_data
    
    :param today: Description
    """
    try:
        data = pd.read_csv("data/btc_ohlcv.csv", index_col=0)
        data.index = pd.to_datetime(data.index)

        today_9 = today.replace(hour=9, minute=0, second=0, microsecond=0)
        time_diff = (today_9.date() - data.index[-1].date()).days
        if time_diff > 0 and today > today_9:
            data = pd.concat([data, pyupbit.get_ohlcv("KRW-BTC", interval="day", count=time_diff)])
            data.to_csv("data/btc_ohlcv.csv")

    except FileNotFoundError:
        data = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=3500)
        data.to_csv("data/btc_ohlcv.csv")
        
    return data
    
def compute_rsi(series, window=14):
    """
    Compute the Relative Strength Index (RSI) for a given price series.

    이 함수는 가격 시계열(series)을 입력으로 받아
    상대강도지수(RSI, Relative Strength Index)를 계산합니다.
    RSI는 일정 기간 동안의 평균 상승폭(average gain)과
    평균 하락폭(average loss)을 비교하여,
    현재 시장의 모멘텀(momentum)과 과매수/과매도 상태를 나타내는
    대표적인 기술적 지표(technical indicator)입니다.

    Parameters
    ----------
    series : pandas.Series
        종가(close price)와 같은 가격 시계열 데이터입니다.
    window : int, default=14
        RSI 계산에 사용되는 이동 평균 기간(rolling window)입니다.
        일반적으로 14일이 표준으로 사용됩니다.

    Returns
    -------
    rsi : pandas.Series
        0부터 100 사이의 값을 가지는 RSI 시계열을 반환합니다.
        일반적으로 RSI 값이 70 이상이면 과매수(overbought),
        30 이하이면 과매도(oversold) 상태로 해석됩니다.

    Notes
    -----
    본 프로젝트에서는 RSI를 개별 시점의 요약 특징(summary feature)으로 사용하여,
    수익률 기반 패턴(return-based pattern)에 시장 모멘텀 정보를 추가하는 데 활용합니다.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_bollinger(series, window=20, k=2):
    """
    Compute Bollinger Bands derived features for a given price series.

    이 함수는 가격 시계열(series)을 입력으로 받아
    볼린저 밴드(Bollinger Bands)를 계산하고,
    그로부터 파생된 두 가지 지표인
    %B (percent_b)와 밴드 폭(Bandwidth)을 반환합니다.

    Bollinger Bands는 이동평균(moving average)을 중심으로
    표준편차(standard deviation)를 이용해 상·하단 밴드를 구성하며,
    가격의 상대적 위치와 변동성(volatility) 상태를 나타내는
    대표적인 변동성 기반 기술적 지표(technical indicator)입니다.

    Parameters
    ----------
    series : pandas.Series
        종가(close price)와 같은 가격 시계열 데이터입니다.
    window : int, default=20
        이동평균과 표준편차를 계산하는 데 사용되는 기간(rolling window)입니다.
        일반적으로 20일이 표준으로 사용됩니다.
    k : int or float, default=2
        상·하단 밴드를 결정하는 표준편차 배수(multiplier)입니다.
        일반적으로 2를 사용합니다.

    Returns
    -------
    percent_b : pandas.Series
        가격이 볼린저 밴드 내에서 차지하는 상대적 위치를 나타내는 지표입니다.
        0에 가까울수록 하단 밴드에 근접하며,
        1에 가까울수록 상단 밴드에 근접함을 의미합니다.

    bandwidth : pandas.Series
        볼린저 밴드의 폭을 이동평균으로 정규화한 값으로,
        시장의 변동성 수준(volatility regime)을 나타냅니다.
        값이 클수록 변동성이 높은 상태를 의미합니다.

    Notes
    -----
    본 프로젝트에서는 Bollinger Bands를 직접 사용하기보다,
    %B와 Bandwidth를 요약 특징(summary features)으로 활용하여
    가격 수익률 패턴에 시장의 변동성 상태 정보를 추가하는 데 사용합니다.
    """
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()

    upper = ma + k * std
    lower = ma - k * std

    percent_b = (series - lower) / (upper - lower)
    bandwidth = (upper - lower) / ma

    return percent_b, bandwidth

def preprocess_btc_data(data, future_days=1):
    try:
        data = data.drop(["open", "high", "low", "volume"], axis=1)

        data["value_change"] = data["value"].pct_change()
        data["close_change"] = data["close"].pct_change()
        
        future_return = data["close"].pct_change(periods=future_days).shift(-future_days)
        data["target_up"] = (future_return > 0).astype("Int64")  # nullable int
        data.loc[future_return.isna(), "target_up"] = pd.NA

        data["rsi"] = compute_rsi(data['close'])
        percent_b, bandwidth = compute_bollinger(data['close'])
        data["bollinger_%b"] = percent_b
        data["bollinger_bandwidth"] = bandwidth

        data = data.drop(["value", "close"], axis=1)
        data.loc[data.index[-future_days:], "target_up"] = pd.NA
    except:
        pass

    return data

def run_knn(data):
    today_data = data.drop("target_up", axis=1).tail(1)
    data = data.dropna()
    x = data.drop("target_up", axis=1)
    y = data["target_up"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2025)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    st.write("train size:", x_train.shape[0])
    st.write("test size:", x_test.shape[0])

    # Create and train the KNN classifier
    k = st.slider("Select number of neighbors (`k`)", min_value=1, max_value=80, value=30)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)

    # Make predictions
    y_pred = knn.predict(x_test)

    # Calculate scores
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    st.write("#### Today's Data for Prediction")
    st.write(today_data)
    today_pred = knn.predict(scaler.transform(today_data))
    

    st.write(f"#### Accuracy: :red[{accuracy:.2f}]")
    st.write(f"#### Balanced Accuracy: :red[{balanced_accuracy:.2f}]")
    st.write(f"#### F1 Score: :red[{f1:.2f}]")
    st.write(f"#### Recall: :red[{recall:.2f}]")
    if today_pred[0] == 1:
        st.write(f"## Today's Prediction: *:green[Up]*")
    else:
        st.write(f"## Today's Prediction: *:red[Down]*")

if __name__ == "__main__":
    st.write("# KNN For :red[BTC Prediction]")
    today = datetime.now()
    st.write(f"Now: `{today.strftime('%Y-%m-%d %H:%M:%S')}`")

    if "loaded" not in st.session_state:
        st.session_state["loaded"] = False
    if "data" not in st.session_state:
        st.session_state["data"] = None
    if "future_days" not in st.session_state:
        st.session_state["future_days"] = 1

    # 며칠 뒤를 예측할지 선택
    future_days = st.slider(
        "Choose how many days in the future you want to predict",
        1, 30,
        key="future_days"
    )

    # 버튼은 트리거만
    if st.button("Load BTC OHLCV Data"):
        st.session_state["data"] = load_btc_data(today)
        st.write(st.session_state["data"])
        st.session_state["loaded"] = True

        data = preprocess_btc_data(
            st.session_state["data"],
            st.session_state["future_days"]
        )
        st.session_state["data"] = data


    # 상태가 유지되는 영역
    if st.session_state["loaded"]:
        st.write("## Preprocessed Data")
        st.write(st.session_state["data"])
        
        try:
            st.write("## Run KNN for BTC Prediction")
            run_knn(st.session_state["data"])
        except NameError:
            st.warning("Please load the data first!")