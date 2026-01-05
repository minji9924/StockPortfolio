import streamlit as st
import pandas as pd
import yfinance as yf
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# --- 상수 및 설정 ---
DATA_FILE = 'my_portfolio_v5.csv'  # 데이터 구조는 v5와 동일
HISTORY_FILE = 'my_history_v5.csv'

# 계좌별 월 고정 입금액
MONTHLY_BUDGET = {
    "ISA": 900000,
    "개인연금": 300000,
    "DC연금": 330000
}

ALL_ACCOUNTS = ["ISA", "개인연금", "DC연금"]

# 종목 정보
TARGET_PORTFOLIO = {
    "105190.KS": {"name": "ACE 200", "ratio": 16.875, "buy_accounts": ["ISA"]},
    "411060.KS": {"name": "ACE KRX금현물", "ratio": 19.000, "buy_accounts": ["개인연금"]},
    "365780.KS": {"name": "ACE 국고채10년", "ratio": 14.250, "buy_accounts": ["DC연금"]},
    "283580.KS": {"name": "KODEX 차이나CSI300", "ratio": 11.875, "buy_accounts": ["ISA"]},
    "360750.KS": {"name": "TIGER 미국S&P500", "ratio": 23.750, "buy_accounts": ["ISA"]},
    "329750.KS": {"name": "TIGER 미국달러단기채권액티브", "ratio": 14.250, "buy_accounts": ["ISA", "DC연금"]},
}

st.set_page_config(page_title="ISA/연금 포트폴리오 프로 (신규입금 배분)", layout="wide")


# --- 데이터 관리 함수 ---
def load_data():
    full_combinations = []
    for ticker, info in TARGET_PORTFOLIO.items():
        for acc in ALL_ACCOUNTS:
            full_combinations.append({'Ticker': ticker, 'Name': info['name'], 'Account': acc})

    base_df = pd.DataFrame(full_combinations)

    if os.path.exists(DATA_FILE):
        saved_df = pd.read_csv(DATA_FILE)
        cols_to_use = ['Ticker', 'Account', 'Shares', 'Total_Cost']
        if 'Total_Cost' not in saved_df.columns:
            saved_df['Total_Cost'] = 0

        merged_df = pd.merge(base_df, saved_df[cols_to_use], on=['Ticker', 'Account'], how='left')
        merged_df['Shares'] = merged_df['Shares'].fillna(0)
        merged_df['Total_Cost'] = merged_df['Total_Cost'].fillna(0)
        return merged_df
    else:
        base_df['Shares'] = 0
        base_df['Total_Cost'] = 0
        base_df.to_csv(DATA_FILE, index=False)
        return base_df


def save_data(df):
    df.to_csv(DATA_FILE, index=False)


def load_history():
    if not os.path.exists(HISTORY_FILE):
        df = pd.DataFrame(columns=['Date', 'Total_Value', 'Total_Cost', 'Return_Rate'])
        df.to_csv(HISTORY_FILE, index=False)
    return pd.read_csv(HISTORY_FILE)


def save_history(record):
    df = load_history()
    current_month = record['Date'][:7]
    df['Month'] = df['Date'].apply(lambda x: x[:7] if isinstance(x, str) else str(x)[:7])

    if current_month in df['Month'].values:
        df = df[df['Month'] != current_month]

    if 'Month' in df.columns:
        df = df.drop(columns=['Month'])

    new_df = pd.DataFrame([record])
    df = pd.concat([df, new_df], ignore_index=True)
    df = df.fillna(0)
    df = df.sort_values('Date')
    df.to_csv(HISTORY_FILE, index=False)


def get_current_prices(tickers):
    if not tickers: return {}
    try:
        data = yf.Tickers(" ".join(tickers))
        prices = {}
        for ticker in tickers:
            try:
                price = data.tickers[ticker].fast_info['last_price']
            except:
                hist = data.tickers[ticker].history(period="1d")
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
                else:
                    price = 0
            prices[ticker] = price
        return prices
    except:
        return {t: 0 for t in tickers}


# --- [변경됨] 매수 계산 로직: 과거 보유량 무시, 신규 자금만 배분 ---
def calculate_buy_plan(prices):
    # 1. 이번 달 총 입금액 계산
    total_new_deposit = sum(MONTHLY_BUDGET.values())

    # 2. 보유 잔고 무시하고, 오직 '총 입금액'에 대한 목표 금액 계산
    buy_needs = []
    for ticker, info in TARGET_PORTFOLIO.items():
        # 과거 기록(df_holdings)을 빼지 않고, 순수하게 비율만 곱함
        target_amt_for_this_month = total_new_deposit * (info['ratio'] / 100.0)

        buy_needs.append({
            'Ticker': ticker,
            'Price': prices[ticker],
            'Needed_Amt': target_amt_for_this_month,  # 리밸런싱 아님. 순수 배분.
            'Buy_Accounts': info['buy_accounts']
        })

    buy_needs_df = pd.DataFrame(buy_needs).set_index('Ticker')

    plan = []
    remaining_budget = MONTHLY_BUDGET.copy()

    # --- Waterfall 배정 로직 (계좌 제약조건 적용) ---

    # (1) 개인연금 - ACE KRX금현물
    t_pension = "411060.KS"
    if buy_needs_df.loc[t_pension, 'Needed_Amt'] > 0:
        p_price = prices[t_pension]
        # 필요 금액과 개인연금 잔고 중 작은 것 선택
        can_buy = min(buy_needs_df.loc[t_pension, 'Needed_Amt'], remaining_budget["개인연금"])
        qty = int(can_buy // p_price)
        cost = qty * p_price
        if qty > 0:
            plan.append(
                {'Ticker': t_pension, 'Account': "개인연금", 'Current_Price': p_price, 'Buy_Qty': qty, 'Cost': cost})
            remaining_budget["개인연금"] -= cost
            buy_needs_df.loc[t_pension, 'Needed_Amt'] -= cost

    # (2) DC연금
    # 2-1. DC 전용 - ACE 국고채10년
    t_dc1 = "365780.KS"
    if buy_needs_df.loc[t_dc1, 'Needed_Amt'] > 0:
        p_price = prices[t_dc1]
        can_buy = min(buy_needs_df.loc[t_dc1, 'Needed_Amt'], remaining_budget["DC연금"])
        qty = int(can_buy // p_price)
        cost = qty * p_price
        if qty > 0:
            plan.append({'Ticker': t_dc1, 'Account': "DC연금", 'Current_Price': p_price, 'Buy_Qty': qty, 'Cost': cost})
            remaining_budget["DC연금"] -= cost
            buy_needs_df.loc[t_dc1, 'Needed_Amt'] -= cost

    # 2-2. DC 잔여로 달러채권
    t_us_bond = "329750.KS"
    needed_bond = buy_needs_df.loc[t_us_bond, 'Needed_Amt']
    if needed_bond > 0 and remaining_budget["DC연금"] > prices[t_us_bond]:
        p_price = prices[t_us_bond]
        can_buy = min(needed_bond, remaining_budget["DC연금"])
        qty = int(can_buy // p_price)
        cost = qty * p_price
        if qty > 0:
            plan.append(
                {'Ticker': t_us_bond, 'Account': "DC연금", 'Current_Price': p_price, 'Buy_Qty': qty, 'Cost': cost})
            remaining_budget["DC연금"] -= cost
            buy_needs_df.loc[t_us_bond, 'Needed_Amt'] -= cost

    # (3) ISA - 나머지 모든 종목 배분
    # 여기서는 남은 필요 금액(Needed_Amt)들의 비율에 따라 ISA 잔고를 나눕니다.
    isa_tickers = ["105190.KS", "283580.KS", "360750.KS", "329750.KS"]

    # ISA에서 사야 할 종목들의 남은 필요 금액 합계
    total_needed_isa = buy_needs_df.loc[isa_tickers, 'Needed_Amt'].sum()
    budget_isa = remaining_budget["ISA"]

    if total_needed_isa > 0:
        for t in isa_tickers:
            needed = buy_needs_df.loc[t, 'Needed_Amt']
            if needed <= 0: continue

            # 남은 필요 금액 비중에 맞춰 ISA 예산 할당
            weight = needed / total_needed_isa
            alloc = budget_isa * weight
            p_price = prices[t]

            qty = int(alloc // p_price)
            cost = qty * p_price

            if qty > 0:
                plan.append({'Ticker': t, 'Account': "ISA", 'Current_Price': p_price, 'Buy_Qty': qty, 'Cost': cost})
                remaining_budget["ISA"] -= cost

    plan_df = pd.DataFrame(plan)
    if not plan_df.empty:
        plan_df['Name'] = plan_df['Ticker'].map(lambda x: TARGET_PORTFOLIO[x]['name'])
    return plan_df, remaining_budget


# ================= UI 구성 =================
st.title("💰 ISA / 연금 포트폴리오 관리자 (v6)")

# ----------------- 사이드바 (데이터 입력) -----------------
st.sidebar.header("🛠 보유 잔고 수정")
st.sidebar.info("보유 수량과 평단가를 관리합니다.")

df_holdings = load_data()
updated_rows = []

for account in ALL_ACCOUNTS:
    with st.sidebar.expander(f"{account} 보유 내역", expanded=False):
        account_rows = df_holdings[df_holdings['Account'] == account]
        for ticker in TARGET_PORTFOLIO.keys():
            curr_row = account_rows[account_rows['Ticker'] == ticker]
            if curr_row.empty:
                row_dict = {'Ticker': ticker, 'Name': TARGET_PORTFOLIO[ticker]['name'], 'Account': account, 'Shares': 0,
                            'Total_Cost': 0}
            else:
                row_dict = curr_row.iloc[0].to_dict()

            st.markdown(f"**{row_dict['Name']}**")
            col_s, col_p = st.columns(2)

            val_shares = col_s.number_input(f"수량", min_value=0, value=int(row_dict['Shares']), step=1,
                                            key=f"shares_{ticker}_{account}")

            current_avg_price = 0
            if row_dict['Shares'] > 0:
                current_avg_price = int(row_dict['Total_Cost'] / row_dict['Shares'])

            val_avg_price = col_p.number_input(f"평단가", min_value=0, value=current_avg_price, step=10,
                                               key=f"price_{ticker}_{account}")

            new_row = row_dict.copy()
            new_row['Shares'] = val_shares
            new_row['Total_Cost'] = val_shares * val_avg_price
            updated_rows.append(new_row)

if st.sidebar.button("잔고 데이터 저장"):
    new_df = pd.DataFrame(updated_rows)
    save_data(new_df)
    st.sidebar.success("저장되었습니다! 화면이 새로고침 됩니다.")
    st.rerun()

df_holdings = pd.DataFrame(updated_rows)

# ----------------- 메인 탭 -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🛒 매수 계획", "📊 자산 현황", "📈 자산 추이", "⚡️ 매수 실행", "🍰 종목 비중 추이"])

with st.spinner("실시간 주가 조회 중..."):
    current_prices = get_current_prices(list(TARGET_PORTFOLIO.keys()))

# --- Tab 1: 매수 계획 (수정됨: 과거 기록 미반영) ---
with tab1:
    st.header("이번 달 매수 가이드")
    st.info("⚠️ 현재 보유 잔고와 무관하게, 이번 달 입금액을 목표 비율대로만 배분합니다.")

    if st.button("계산하기", type="primary"):
        # calculate_buy_plan에 df_holdings를 넘기지 않음
        plan_df, remain_cash = calculate_buy_plan(current_prices)
        if not plan_df.empty:
            for acct in ALL_ACCOUNTS:
                st.subheader(f"{acct}")
                acct_plan = plan_df[plan_df['Account'] == acct]
                if not acct_plan.empty:
                    display_df = acct_plan[['Name', 'Current_Price', 'Buy_Qty', 'Cost']].copy()
                    display_df.columns = ['종목명', '현재가', '매수 수량', '예상 금액']
                    st.dataframe(
                        display_df.style.format({
                            '현재가': '{:,.0f}',
                            '예상 금액': '{:,.0f}'
                        }),
                        hide_index=True,
                        use_container_width=True
                    )
                    st.info(f"잔여 현금: {remain_cash.get(acct, 0):,.0f}원")
                else:
                    st.caption("매수할 종목 없음")
        else:
            st.warning("매수할 수량이 없습니다.")

# --- Tab 2: 수익률 현황 ---
with tab2:
    st.header("자산 및 수익률 현황")

    df_calc = df_holdings.copy()
    df_calc['Current_Price'] = df_calc['Ticker'].map(current_prices)
    df_calc['Eval_Value'] = df_calc['Shares'] * df_calc['Current_Price']
    df_calc['Profit'] = df_calc['Eval_Value'] - df_calc['Total_Cost']
    df_calc['Yield'] = df_calc.apply(lambda x: (x['Profit'] / x['Total_Cost'] * 100) if x['Total_Cost'] > 0 else 0,
                                     axis=1)

    # 전체 요약
    total_invest = df_calc['Total_Cost'].sum()
    total_eval = df_calc['Eval_Value'].sum()
    total_profit = total_eval - total_invest
    total_yield = (total_profit / total_invest * 100) if total_invest > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 평가금액", f"{total_eval:,.0f}원")
    col2.metric("총 매수금액", f"{total_invest:,.0f}원")
    col3.metric("총 평가손익", f"{total_profit:,.0f}원", delta_color="normal")
    col4.metric("총 수익률", f"{total_yield:.2f}%", delta=f"{total_yield:.2f}%")

    st.divider()

    # 상세 테이블
    df_view = df_calc[df_calc['Shares'] > 0].copy()
    if not df_view.empty:
        df_view['Avg_Price'] = df_view['Total_Cost'] / df_view['Shares']
        display_cols = ['Account', 'Name', 'Shares', 'Avg_Price', 'Current_Price', 'Total_Cost', 'Eval_Value', 'Yield']
        rename_map = {'Account': '계좌', 'Name': '종목명', 'Shares': '보유수량', 'Avg_Price': '평단가',
                      'Current_Price': '현재가', 'Total_Cost': '매수금액', 'Eval_Value': '평가금액', 'Yield': '수익률(%)'}
        final_view = df_view[display_cols].rename(columns=rename_map)
        st.dataframe(final_view.style.format({'평단가': '{:,.0f}', '현재가': '{:,.0f}', '매수금액': '{:,.0f}',
                                              '평가금액': '{:,.0f}', '수익률(%)': '{:.2f}'})
                     .map(lambda x: 'color: red' if x > 0 else 'color: blue', subset=['수익률(%)']),
                     use_container_width=True)
    else:
        st.info("보유 주식이 없습니다.")

    st.divider()
    if st.button("현재 자산 상태 및 종목별 비중 저장"):
        today = datetime.now().strftime("%Y-%m-%d")
        record = {
            'Date': today,
            'Total_Value': total_eval,
            'Total_Cost': total_invest,
            'Return_Rate': total_yield
        }
        ticker_groups = df_calc.groupby('Ticker')['Eval_Value'].sum()
        for ticker, val in ticker_groups.items():
            record[ticker] = val

        save_history(record)
        st.success("기록 완료! '종목 비중 추이' 탭에서 데이터를 확인할 수 있습니다.")

# --- Tab 3: 자산 추이 ---
with tab3:
    st.header("자산 성장 & 수익률 추이")
    hist_df = load_history()
    if not hist_df.empty:
        hist_df['Date'] = pd.to_datetime(hist_df['Date'])

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['Total_Value'], name='총 평가액', line=dict(color='red')))
        fig1.add_trace(
            go.Scatter(x=hist_df['Date'], y=hist_df['Total_Cost'], name='총 매수원금', line=dict(color='gray', dash='dot')))
        fig1.update_layout(title="자산 성장", xaxis_title="날짜", yaxis_title="금액(원)")
        st.plotly_chart(fig1, use_container_width=True)

        if 'Return_Rate' in hist_df.columns:
            fig2 = px.line(hist_df, x='Date', y='Return_Rate', title="누적 수익률 변화 (%)", markers=True)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("저장된 기록이 없습니다.")

# --- Tab 4: 매수 실행 ---
with tab4:
    st.header("⚡️ 매수 실행 및 잔고 업데이트")
    if 'buy_plan_df' not in st.session_state: st.session_state.buy_plan_df = pd.DataFrame()

    if st.button("매수 계획 불러오기"):
        # 여기서도 과거 기록 미반영 로직 사용
        plan_df, _ = calculate_buy_plan(current_prices)
        if not plan_df.empty:
            edit_df = plan_df[['Ticker', 'Name', 'Account', 'Buy_Qty', 'Current_Price']].copy()
            edit_df.columns = ['Ticker', '종목명', '계좌', '매수수량', '체결단가']
            st.session_state.buy_plan_df = edit_df
        else:
            st.session_state.buy_plan_df = pd.DataFrame()
            st.warning("매수할 계획이 없습니다.")

    if not st.session_state.buy_plan_df.empty:
        edited_df = st.data_editor(st.session_state.buy_plan_df,
                                   column_config={"매수수량": st.column_config.NumberColumn(min_value=0),
                                                  "체결단가": st.column_config.NumberColumn(min_value=0, format="%d원")},
                                   hide_index=True, num_rows="dynamic")

        if st.button("✅ 실제 잔고에 반영하기"):
            current_holdings = load_data()
            for index, row in edited_df.iterrows():
                mask = (current_holdings['Ticker'] == row['Ticker']) & (current_holdings['Account'] == row['계좌'])
                cost = row['매수수량'] * row['체결단가']
                if mask.any():
                    current_holdings.loc[mask, 'Shares'] += row['매수수량']
                    current_holdings.loc[mask, 'Total_Cost'] += cost
                else:
                    new_row = {'Ticker': row['Ticker'], 'Name': row['종목명'], 'Account': row['계좌'],
                               'Shares': row['매수수량'], 'Total_Cost': cost}
                    current_holdings = pd.concat([current_holdings, pd.DataFrame([new_row])], ignore_index=True)

            save_data(current_holdings)
            st.session_state.buy_plan_df = pd.DataFrame()
            st.success("반영 완료!")
            st.rerun()

# --- Tab 5: 종목 비중 추이 ---
with tab5:
    st.header("🍰 전체 포트폴리오 종목 비중 변화")
    hist_df = load_history()

    if not hist_df.empty:
        hist_df['Date'] = pd.to_datetime(hist_df['Date'])
        meta_cols = ['Date', 'Total_Value', 'Total_Cost', 'Return_Rate']
        ticker_cols = [c for c in hist_df.columns if c not in meta_cols]

        if ticker_cols:
            df_melted = hist_df.melt(id_vars=['Date', 'Total_Value'], value_vars=ticker_cols,
                                     var_name='Ticker', value_name='Value')
            df_melted['Name'] = df_melted['Ticker'].map(lambda x: TARGET_PORTFOLIO.get(x, {}).get('name', x))
            df_melted['Percentage'] = df_melted.apply(
                lambda row: (row['Value'] / row['Total_Value'] * 100) if row['Total_Value'] > 0 else 0,
                axis=1
            )

            fig = px.area(df_melted, x='Date', y='Percentage', color='Name',
                          title="시간 흐름에 따른 종목 비중 변화 (%)",
                          labels={'Percentage': '비중 (%)'},
                          groupnorm=None)
            fig.update_yaxes(range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("상세 데이터")
            st.dataframe(hist_df)
        else:
            st.info("아직 저장된 종목별 데이터가 없습니다. '자산 현황' 탭에서 기록을 저장해주세요.")
    else:
        st.info("기록된 데이터가 없습니다.")