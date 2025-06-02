#%%
import pandas as pd
import numpy as np
import scipy.optimize as sco # [新增] 為了最小變異數優化 (雖然此版本未使用，但保留以備不時之需)
import xgboost as xgb # 保留，但未使用
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.metrics import r2_score
# from sklearn.preprocessing import StandardScaler # 此版本未使用正規化
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick # 用於格式化Y軸為百分比
import warnings

# ==============================================================================
# 組態設定 (Config)
# ==============================================================================
class Config:
    # --- 檔案路徑 ---
    FEATURE_FILE = "standardized_feature.xlsx" # 因子數據檔案
    PRICE_FILE = "filtered_price.xlsx"      # 價格數據檔案
    INDEX_FILE = "e_index.xlsx"         # [新增] 電子業指數數據檔案
    
    # --- 模型與特徵 ---
    FACTOR_COLS = ['PE_Q', 'PB_Q', 'PSR_Q', 'Dividend_Yield', 'EBITDA', 'Revenue_Growth',
                   'Asset_Growth', 'Equity_Growth', 'Debt_Ratio', 'ROA', 'ROE',
                   'NetIncome_Growth', 'PEG', 'Momentum', 'Rolling_1Y_Std'] # 使用的因子欄位
    TARGET_COL = 'Next_Quarterly_Return' # 預測目標：下一季報酬率
    TRAIN_LIMIT = '2023-10-01' # 訓練集數據截止日期
    
    # --- 超參數調校 (RandomForestRegressor) ---
    PARAM_GRID = {
        'n_estimators': [100], 
        'max_depth': [3, 5], 
        'min_samples_leaf': [20, 50], 
        'max_features': ['sqrt']
    } # RandomForestRegressor的超參數搜尋範圍
    
    # --- 回測設定 (高階版) ---
    BACKTEST_START_DATE = '2024-01-01' # 回測開始日期
    INITIAL_CAPITAL = 100000.00      # 初始資金
    TRANSACTION_COST_RATE = 0.003    # 交易成本率 (買賣各算一次)
    SLIPPAGE_RATE = 0.001            # 滑價率 (買入價上浮，賣出價下浮)

    MIN_PREDICTED_RETURN = 0.0  # 預測分數過濾: 僅選擇模型預測季報酬率 > 0 的股票
    MAX_STOCKS_TO_HOLD = 20     # 動態持股數量的上限 (最多持有幾支股票)

    USE_TRAILING_STOP = True      # 是否啟用移動停損
    STOP_LOSS_THRESHOLD = -0.20   # 移動停損: 從波段高點回撤 20% 則停損
    TAKE_PROFIT_THRESHOLD = 0.40  # 停利閾值: 從買入價上漲 40% 則停利

    WEIGHTING_STRATEGY = 'MIN_VARIANCE' # 權重分配策略: 'EQUAL_WEIGHT' (等權重) 或 'MIN_VARIANCE' (最小變異數)
    COVARIANCE_WINDOW = 30              # 計算共變異數矩陣所需的回看天數 (交易日)

    REBALANCE_DAYS = 3 # 將再平衡交易分散到 3 天內完成，以平滑交易價格

# --- 初始化設定 ---
cfg = Config()
warnings.filterwarnings('ignore') # 忽略警告訊息

# 修正中文字體顯示問題
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 設定圖表使用微軟正黑體
    plt.rcParams['axes.unicode_minus'] = False # 解決圖表負號顯示問題
except Exception as e:
    print(f"設定中文字體失敗: {e}. 圖表中的中文可能無法正常顯示。")

# ==============================================================================
# 1. 數據加載與預處理
# ==============================================================================
print("步驟 1: 數據加載與預處理...")
try:
    data = pd.read_excel(cfg.FEATURE_FILE)
    price_data = pd.read_excel(cfg.PRICE_FILE)
except FileNotFoundError as e:
    print(f"錯誤：檔案未找到 {e}。請確認 '{cfg.FEATURE_FILE}' 和 '{cfg.PRICE_FILE}' 檔案位於正確的路徑。")
    exit()

print("正在將 StockID 型別統一為整數...")
data['StockID'] = data['StockID'].astype(int)
price_data['StockID'] = price_data['StockID'].astype(str).str.extract(r'(^\d+)').astype(int)

# 預處理價格檔案
price_data['Date'] = pd.to_datetime(price_data['Date']).dt.normalize()
price_data = price_data.rename(columns={'Close': 'Price', 'close': 'Price'}) # 將 'Close' 或 'close' 改名為 'Price'
price_data = price_data.drop_duplicates(subset=['Date', 'StockID'], keep='first') # 移除重複數據
daily_prices = price_data.set_index(['Date', 'StockID']).sort_index() # 設定索引並排序

# 預處理因子檔案
data['Date'] = pd.to_datetime(data['Date']).dt.normalize()
data = data.set_index(['StockID', 'Date']).sort_index() # 設定索引並排序

# 計算目標變數：下一季的報酬率
data[cfg.TARGET_COL] = data.groupby(level='StockID')['Quarterly_Return'].shift(-1)
data = data.dropna(subset=[cfg.TARGET_COL]) # 移除目標變數為空的數據
data = data.fillna(0) # 其餘空值填0 (可根據實際情況調整處理方式)
print("數據預處理完成。")

# ==============================================================================
# 2. 特徵工程與數據分割 (此版本未使用正規化)
# ==============================================================================
print("\n步驟 2: 特徵工程與數據分割...")
X = data[cfg.FACTOR_COLS].apply(pd.to_numeric, errors='coerce').fillna(0) # 選取因子欄位並轉為數值
y = data[cfg.TARGET_COL] # 選取目標欄位

# 根據日期分割訓練集與測試集
train_idx = data.index.get_level_values('Date') < cfg.TRAIN_LIMIT
test_idx = data.index.get_level_values('Date') >= cfg.TRAIN_LIMIT

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
print(f"訓練集大小: {X_train.shape[0]}, 測試集大小: {X_test.shape[0]}")

# ==============================================================================
# 3. 超參數調校 (GridSearchCV) - 使用RandomForestRegressor
# ==============================================================================
print("\n步驟 3: 超參數調校 (GridSearchCV with RandomForestRegressor)...")
rf_model_for_grid = RandomForestRegressor(random_state=42, n_jobs=-1) # 初始化隨機森林模型
tscv = TimeSeriesSplit(n_splits=5) # 使用時間序列交叉驗證
grid_search = GridSearchCV(estimator=rf_model_for_grid, 
                           param_grid=cfg.PARAM_GRID, 
                           cv=tscv, 
                           scoring='neg_mean_squared_error', 
                           verbose=1, # 顯示進度
                           n_jobs=-1)
grid_search.fit(X_train, y_train) # 執行超參數搜尋
print(f"找到的最佳參數: {grid_search.best_params_}")
best_rf_model_from_grid = grid_search.best_estimator_ # 取得最佳模型配置

# ==============================================================================
# 4. 因子篩選 (RFECV) - 使用最佳RandomForest模型
# ==============================================================================
print("\n步驟 4: 因子篩選 (RFECV)...")
selector = RFECV(estimator=best_rf_model_from_grid, 
                 step=1, 
                 cv=tscv, 
                 scoring='neg_mean_squared_error', 
                 n_jobs=-1, 
                 min_features_to_select=3) 
selector.fit(X_train, y_train) 
selected_features = X_train.columns[selector.support_].tolist() 
print(f"RFECV 篩選後保留 {selector.n_features_} 個因子: {selected_features}")

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# ==============================================================================
# 5. 訓練最終模型與評估 - 使用篩選後的特徵
# ==============================================================================
print("\n步驟 5: 訓練最終模型與評估...")
final_model = RandomForestRegressor(**grid_search.best_params_, random_state=42, n_jobs=-1)
final_model.fit(X_train_selected, y_train) 

y_train_pred = final_model.predict(X_train_selected)
y_test_pred = final_model.predict(X_test_selected)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
print(f"\n訓練集 R²: {r2_train:.4f}")
print(f"測試集 R²: {r2_test:.4f}")

if r2_train > r2_test + 0.3:
    print(f"警告：訓練集R²({r2_train:.3f})與測試集R²({r2_test:.3f})差異過大，可能仍有過擬合。")
elif r2_test < 0.01: 
    print(f"警告：測試集R²({r2_test:.3f})過低，模型預測能力可能不足。")
else:
    print("模型泛化能力看起來較為合理。")

# ==============================================================================
# 6. 特徵重要性分析
# ==============================================================================
print("\n步驟 6: 特徵重要性分析 (最終模型)...")
importances = final_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print("最終模型特徵重要性排名：")
print(feature_importance_df)

plt.figure(figsize=(10, 6))
plt.title('最終模型特徵重要性 (RandomForest)')
plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xticks(rotation=45, ha="right")
plt.ylabel('重要性分數')
plt.tight_layout()
plt.show()

#%%
# ==============================================================================
# 7. 詳細每日回測 (高階優化版)
# ==============================================================================
print("\n步驟 7: 詳細每日回測 (高階優化版)...")

test_data_backtest = data[test_idx].copy() 
test_data_backtest['Predicted_Return'] = y_test_pred 
test_data_backtest = test_data_backtest.sort_index(level='Date')
print("已將測試數據按日期排序以進行回測。")

def get_minimum_variance_weights(target_stocks_list, current_date, cov_window, prices_df):
    if not target_stocks_list: return None
    start_date = current_date - pd.Timedelta(days=cov_window * 2.5) 
    relevant_prices_df = prices_df[prices_df.index.get_level_values('StockID').isin(target_stocks_list)]
    hist_prices_unstacked = relevant_prices_df.loc[(slice(start_date, current_date - pd.Timedelta(days=1))), 'Price'].unstack(level='StockID')
    hist_prices_unstacked = hist_prices_unstacked.reindex(columns=target_stocks_list)
    hist_prices_unstacked = hist_prices_unstacked.dropna(axis=0, how='all').dropna(axis=1, how='any') 
    valid_target_stocks = hist_prices_unstacked.columns.tolist()
    if len(valid_target_stocks) < 2: return None
    returns = hist_prices_unstacked.pct_change().dropna()
    if len(returns) < cov_window * 0.5: return None
    cov_matrix = returns.cov() * 252 
    num_assets = len(valid_target_stocks)
    args = (cov_matrix,)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) 
    bounds = tuple((0, 1) for _ in range(num_assets)) 
    initial_guess = num_assets * [1. / num_assets] 
    def portfolio_variance(weights, cov_matrix_arg): return np.dot(weights.T, np.dot(cov_matrix_arg, weights))
    result = sco.minimize(portfolio_variance, initial_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success: return None
    return pd.Series(result.x, index=valid_target_stocks)

cash = cfg.INITIAL_CAPITAL
portfolio = {} 
portfolio_history = []
daily_log = [] 
trading_days = daily_prices.loc[cfg.BACKTEST_START_DATE:].index.get_level_values('Date').unique().sort_values()
rebalance_state = {'needs_rebalance': False, 'target_portfolio_weights': {}, 'rebalance_days_left': 0}

for day in trading_days:
    day_str = day.strftime('%Y-%m-%d')
    daily_log.append(f"\n--- 日期: {day_str} ---")
    daily_log.append(f"本日開始現金: ${cash:,.2f}")

    current_value_of_stocks = 0
    for stock_id, details in portfolio.items():
        if (day, stock_id) in daily_prices.index:
            current_stock_price = daily_prices.loc[(day, stock_id), 'Price']
            details['last_price'] = current_stock_price 
            current_value_of_stocks += details['shares'] * current_stock_price
        elif 'last_price' in details: 
             current_value_of_stocks += details['shares'] * details['last_price']
    current_total_portfolio_value = cash + current_value_of_stocks
    daily_log.append(f"本日開始總資產: ${current_total_portfolio_value:,.2f} (股票市值: ${current_value_of_stocks:,.2f})")

    current_quarter_period = pd.Period(day, freq='Q')
    first_trading_day_of_this_quarter = trading_days[trading_days.to_period('Q') == current_quarter_period][0]

    if day == first_trading_day_of_this_quarter and not rebalance_state['needs_rebalance']:
        log_msg = f"\n===== 於 {day_str} 觸發 {str(current_quarter_period)} 季度再平衡程序 ====="
        print(log_msg); daily_log.append(log_msg)
        feature_date_target = day - pd.DateOffset(days=1) 
        unique_feature_dates = pd.DatetimeIndex(test_data_backtest.index.get_level_values('Date').unique()).sort_values()
        actual_feature_date = unique_feature_dates.asof(feature_date_target)
        selected_target_stocks = []
        if pd.notna(actual_feature_date):
            log_msg = f"使用最新的因子數據日期: {actual_feature_date.strftime('%Y-%m-%d')} 來進行選股"
            print(log_msg); daily_log.append(log_msg)
            current_quarter_factor_data = test_data_backtest[test_data_backtest.index.get_level_values('Date') == actual_feature_date]
            candidates = current_quarter_factor_data[current_quarter_factor_data['Predicted_Return'] > cfg.MIN_PREDICTED_RETURN]
            selected_target_stocks = candidates.nlargest(cfg.MAX_STOCKS_TO_HOLD, 'Predicted_Return').index.get_level_values('StockID').tolist()
        if selected_target_stocks:
            log_msg = f"本季候選目標 ({len(selected_target_stocks)}支): {selected_target_stocks}"
            print(log_msg); daily_log.append(log_msg)
            calculated_weights = None
            if cfg.WEIGHTING_STRATEGY == 'MIN_VARIANCE':
                log_msg = "計算最小變異數權重..."
                print(log_msg); daily_log.append(log_msg)
                calculated_weights = get_minimum_variance_weights(selected_target_stocks, day, cfg.COVARIANCE_WINDOW, daily_prices)
                if calculated_weights is None or calculated_weights.empty:
                    log_msg = "警告: 最小變異數計算失敗或未返回有效權重，本季退回至等權重。"
                    print(log_msg); daily_log.append(log_msg)
            if calculated_weights is None or calculated_weights.empty: 
                log_msg = "採用等權重策略..."
                print(log_msg); daily_log.append(log_msg)
                tradable_target_stocks = [s for s in selected_target_stocks if (day,s) in daily_prices.index]
                if tradable_target_stocks:
                    calculated_weights = pd.Series(1 / len(tradable_target_stocks), index=tradable_target_stocks)
                else:
                    log_msg = "警告: 目標股票均無法在今日交易，本季不建立新倉位。"
                    print(log_msg); daily_log.append(log_msg)
                    calculated_weights = pd.Series(dtype=float)
            rebalance_state['target_portfolio_weights'] = calculated_weights.to_dict() if calculated_weights is not None else {}
            rebalance_state['needs_rebalance'] = True
            rebalance_state['rebalance_days_left'] = cfg.REBALANCE_DAYS
            log_msg = f"目標權重設定完畢，將在 {cfg.REBALANCE_DAYS} 天內完成再平衡。目標權重: {rebalance_state['target_portfolio_weights']}"
            print(log_msg); daily_log.append(log_msg)
        else:
            log_msg = "本季無符合條件的股票，將清空倉位。"
            print(log_msg); daily_log.append(log_msg)
            rebalance_state['target_portfolio_weights'] = {} 
            rebalance_state['needs_rebalance'] = True
            rebalance_state['rebalance_days_left'] = 1 
    if rebalance_state['rebalance_days_left'] > 0 and rebalance_state['needs_rebalance']:
        log_msg = f"--- {day_str}: 執行再平衡 (剩餘 {rebalance_state['rebalance_days_left']} 天) ---"
        print(log_msg); daily_log.append(log_msg)
        target_value_per_stock = { stock_id: current_total_portfolio_value * weight for stock_id, weight in rebalance_state['target_portfolio_weights'].items() }
        current_value_per_stock = {}
        for stock_id, details in portfolio.items():
            if (day, stock_id) in daily_prices.index: current_value_per_stock[stock_id] = details['shares'] * daily_prices.loc[(day, stock_id), 'Price']
            elif 'last_price' in details: current_value_per_stock[stock_id] = details['shares'] * details['last_price']
        all_relevant_stocks = set(target_value_per_stock.keys()) | set(current_value_per_stock.keys())
        for stock_id in all_relevant_stocks:
            target_val = target_value_per_stock.get(stock_id, 0) 
            current_val = current_value_per_stock.get(stock_id, 0)
            value_difference_to_trade = (target_val - current_val) / rebalance_state['rebalance_days_left']
            if (day, stock_id) in daily_prices.index:
                price_today = daily_prices.loc[(day, stock_id), 'Price']
                if value_difference_to_trade > 1: 
                    buy_price_with_slippage = price_today * (1 + cfg.SLIPPAGE_RATE)
                    shares_to_buy = value_difference_to_trade / buy_price_with_slippage
                    value_of_purchase = shares_to_buy * buy_price_with_slippage 
                    cost_of_transaction = value_of_purchase * cfg.TRANSACTION_COST_RATE 
                    total_cash_deducted = value_of_purchase + cost_of_transaction 
                    if cash >= total_cash_deducted : 
                        cash -= total_cash_deducted
                        if stock_id not in portfolio: portfolio[stock_id] = {'shares': 0, 'purchase_price': buy_price_with_slippage, 'peak_price_since_purchase': buy_price_with_slippage, 'last_price': buy_price_with_slippage}
                        if portfolio[stock_id]['shares'] == 0: portfolio[stock_id]['purchase_price'] = buy_price_with_slippage; portfolio[stock_id]['peak_price_since_purchase'] = buy_price_with_slippage
                        portfolio[stock_id]['shares'] += shares_to_buy
                        portfolio[stock_id]['last_price'] = buy_price_with_slippage
                        log_msg = (f"  買入 {stock_id}: {shares_to_buy:.2f} 股 @ 實際買價 ${buy_price_with_slippage:.2f} (含滑價), 買入金額 ${value_of_purchase:,.2f}, 交易成本 ${cost_of_transaction:,.2f}, 總使用資金 ${total_cash_deducted:,.2f}, 剩餘現金 ${cash:,.2f}")
                        print(log_msg); daily_log.append(log_msg)
                elif value_difference_to_trade < -1: 
                    sell_price_with_slippage = price_today * (1 - cfg.SLIPPAGE_RATE)
                    shares_to_sell = abs(value_difference_to_trade) / sell_price_with_slippage
                    if stock_id in portfolio and portfolio[stock_id]['shares'] > 0:
                        shares_actually_sold = min(shares_to_sell, portfolio[stock_id]['shares'])
                        value_of_sale = shares_actually_sold * sell_price_with_slippage 
                        cost_of_transaction = value_of_sale * cfg.TRANSACTION_COST_RATE 
                        total_cash_received = value_of_sale - cost_of_transaction 
                        portfolio[stock_id]['shares'] -= shares_actually_sold
                        cash += total_cash_received
                        log_msg = (f"  賣出 {stock_id}: {shares_actually_sold:.2f} 股 @ 實際賣價 ${sell_price_with_slippage:.2f} (含滑價), 賣出金額 ${value_of_sale:,.2f}, 交易成本 ${cost_of_transaction:,.2f}, 實收現金 ${total_cash_received:,.2f}, 剩餘現金 ${cash:,.2f}")
                        print(log_msg); daily_log.append(log_msg)
                        if portfolio[stock_id]['shares'] < 1e-6: del portfolio[stock_id]
            else:
                log_msg = f"  注意: {stock_id} 今日無價格數據，無法執行再平衡交易。"
                print(log_msg); daily_log.append(log_msg)
        rebalance_state['rebalance_days_left'] -= 1
        if rebalance_state['rebalance_days_left'] == 0:
            rebalance_state['needs_rebalance'] = False
            log_msg = "--- 再平衡完成 ---"; print(log_msg); daily_log.append(log_msg)
            daily_log.append(f"再平衡後現金: ${cash:,.2f}")
    stocks_to_force_sell = [] 
    for stock_id, details in list(portfolio.items()): 
        if (day, stock_id) in daily_prices.index:
            current_price = daily_prices.loc[(day, stock_id), 'Price']
            details['last_price'] = current_price 
            trigger_event = None
            if cfg.USE_TRAILING_STOP:
                details['peak_price_since_purchase'] = max(details.get('peak_price_since_purchase', details['purchase_price']), current_price)
                if current_price <= details['peak_price_since_purchase'] * (1 + cfg.STOP_LOSS_THRESHOLD): trigger_event = '移動停損'
            if current_price >= details['purchase_price'] * (1 + cfg.TAKE_PROFIT_THRESHOLD): trigger_event = '停利'
            if trigger_event: stocks_to_force_sell.append((stock_id, trigger_event))
    if stocks_to_force_sell: daily_log.append(f"--- {day_str}: 執行停損停利檢查 ---")
    for stock_id, event_type in stocks_to_force_sell:
        if stock_id in portfolio: 
            price_at_trigger = portfolio[stock_id]['last_price'] 
            sell_price_with_slippage = price_at_trigger * (1 - cfg.SLIPPAGE_RATE)
            shares_at_trigger = portfolio[stock_id]['shares']
            value_of_sale = shares_at_trigger * sell_price_with_slippage
            cost_of_transaction = value_of_sale * cfg.TRANSACTION_COST_RATE
            total_cash_received = value_of_sale - cost_of_transaction
            cash += total_cash_received
            log_msg = (f"{day_str}: {stock_id} 觸發 {event_type} @ 觸發價 ${price_at_trigger:.2f}, 實際賣價 ${sell_price_with_slippage:.2f} (含滑價), 賣出股數 {shares_at_trigger:.2f}, 賣出金額 ${value_of_sale:,.2f}, 交易成本 ${cost_of_transaction:,.2f}, 實收現金 ${total_cash_received:,.2f}, 剩餘現金 ${cash:,.2f}。清倉。")
            print(log_msg); daily_log.append(log_msg)
            del portfolio[stock_id]
            if stock_id in rebalance_state['target_portfolio_weights']:
                del rebalance_state['target_portfolio_weights'][stock_id]
                log_msg = f"  {stock_id} 已從本輪再平衡目標中移除。"
                print(log_msg); daily_log.append(log_msg)
    if stocks_to_force_sell: daily_log.append(f"停損停利執行後現金: ${cash:,.2f}")
    final_day_portfolio_value = cash
    current_value_of_stocks_eod = 0 
    for stock_id, details in portfolio.items():
        if 'last_price' in details: 
             stock_value = details['shares'] * details['last_price']
             final_day_portfolio_value += stock_value 
             current_value_of_stocks_eod += stock_value
        elif (day, stock_id) in daily_prices.index: 
            stock_price_eod = daily_prices.loc[(day, stock_id), 'Price']
            stock_value = details['shares'] * stock_price_eod
            final_day_portfolio_value += stock_value
            current_value_of_stocks_eod += stock_value
    portfolio_history.append({'Date': day, 'Portfolio_Value': final_day_portfolio_value})
    daily_log.append(f"本日結束總資產: ${final_day_portfolio_value:,.2f} (現金: ${cash:,.2f}, 股票市值: ${current_value_of_stocks_eod:,.2f})")

# ==============================================================================
# 8. 基於每日回測的績效報告 - [指數比較修改]
# ==============================================================================
print("\n步驟 8: 績效報告與比較...")

log_file_name = f"backtest_daily_log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(log_file_name, 'w', encoding='utf-8') as f:
    for log_entry in daily_log:
        f.write(log_entry + '\n')
print(f"每日詳細回測日誌已儲存至: {log_file_name}")

if not portfolio_history:
    print("回測期間無任何交易，無法產生績效報告。")
else:
    report_df = pd.DataFrame(portfolio_history).set_index('Date')
    
    # --- 策略績效計算 ---
    strategy_total_return = (report_df['Portfolio_Value'].iloc[-1] / cfg.INITIAL_CAPITAL) - 1
    num_trading_days_in_backtest = len(report_df)
    
    if num_trading_days_in_backtest > 0:
        strategy_annualized_return = (1 + strategy_total_return) ** (252 / num_trading_days_in_backtest) - 1
    else:
        strategy_annualized_return = 0
    
    report_df['Strategy_Daily_Return'] = report_df['Portfolio_Value'].pct_change().fillna(0)
    strategy_annualized_volatility = report_df['Strategy_Daily_Return'].std() * np.sqrt(252)
    strategy_sharpe_ratio = strategy_annualized_return / strategy_annualized_volatility if strategy_annualized_volatility != 0 else 0

    report_df['Strategy_Cumulative_Return_Factor'] = (1 + report_df['Strategy_Daily_Return']).cumprod() 
    report_df['Strategy_Peak_Value'] = (report_df['Strategy_Cumulative_Return_Factor'] * cfg.INITIAL_CAPITAL).cummax() 
    report_df['Strategy_Drawdown_Value'] = report_df['Portfolio_Value'] - report_df['Strategy_Peak_Value'] 
    report_df['Strategy_Drawdown_Percentage'] = np.where(report_df['Strategy_Peak_Value'] == 0, 0, report_df['Strategy_Drawdown_Value'] / report_df['Strategy_Peak_Value'])
    strategy_max_drawdown = report_df['Strategy_Drawdown_Percentage'].min() if not report_df['Strategy_Drawdown_Percentage'].empty else 0

    print("\n策略績效報告 (每日回測):")
    print("-" * 30)
    print(f"最終投資組合價值: ${report_df['Portfolio_Value'].iloc[-1]:,.2f}")
    print(f"總報酬率: {strategy_total_return:.2%}")
    print(f"年化報酬率: {strategy_annualized_return:.2%}")
    print(f"年化波動率: {strategy_annualized_volatility:.2%}")
    print(f"夏普比率 (Sharpe Ratio): {strategy_sharpe_ratio:.2f}")
    print(f"最大回撤 (Max Drawdown): {strategy_max_drawdown:.2%}")
    print("-" * 30)

    # --- [新增] 電子業指數比較 ---
    try:
        index_data = pd.read_excel(cfg.INDEX_FILE)
        # 標準化日期欄位名稱
        date_col_found = False
        possible_date_cols = ['日期']
        for col_name_variant in possible_date_cols:
            if col_name_variant in index_data.columns:
                index_data = index_data.rename(columns={col_name_variant: 'Date'})
                date_col_found = True
                break
        if not date_col_found:
            raise ValueError(f"指數文件中未找到可識別的日期欄位 (請嘗試 {possible_date_cols})")

        # 標準化價格欄位名稱
        price_col_found = False
        # [修改] 增加 "價格指數" 到可能的價格欄位名列表
        possible_price_cols = ['價格指數值']
        for col_name_variant in possible_price_cols:
            if col_name_variant in index_data.columns:
                index_data = index_data.rename(columns={col_name_variant: 'Index_Price'})
                price_col_found = True
                break
        if not price_col_found:
            raise ValueError(f"指數文件中未找到可識別的價格欄位 (請嘗試 {possible_price_cols})")

        index_data['Date'] = pd.to_datetime(index_data['Date']).dt.normalize()
        index_data = index_data.set_index('Date')
        index_data = index_data[['Index_Price']].dropna() 
        index_data['Index_Price'] = pd.to_numeric(index_data['Index_Price'], errors='coerce') 
        index_data = index_data.dropna(subset=['Index_Price'])

        index_aligned = index_data.reindex(report_df.index).ffill().bfill() 

        if not index_aligned.empty and not index_aligned['Index_Price'].isnull().all():
            report_df['Index_Value_Plot'] = (index_aligned['Index_Price'] / index_aligned['Index_Price'].iloc[0]) * cfg.INITIAL_CAPITAL
            report_df['Index_Daily_Return'] = index_aligned['Index_Price'].pct_change().fillna(0)
            index_total_return = (index_aligned['Index_Price'].iloc[-1] / index_aligned['Index_Price'].iloc[0]) - 1
            
            if num_trading_days_in_backtest > 0:
                index_annualized_return = (1 + index_total_return) ** (252 / num_trading_days_in_backtest) - 1
            else:
                index_annualized_return = 0
            
            index_annualized_volatility = report_df['Index_Daily_Return'].std() * np.sqrt(252)
            index_sharpe_ratio = index_annualized_return / index_annualized_volatility if index_annualized_volatility != 0 else 0

            report_df['Index_Cumulative_Return_Factor'] = (1 + report_df['Index_Daily_Return']).cumprod()
            index_first_valid_price = index_aligned['Index_Price'].dropna().iloc[0] if not index_aligned['Index_Price'].dropna().empty else 1
            report_df['Index_Peak_Value_Raw'] = (report_df['Index_Cumulative_Return_Factor'] * index_first_valid_price).cummax() 
            report_df['Index_Drawdown_Value'] = index_aligned['Index_Price'] - report_df['Index_Peak_Value_Raw']
            report_df['Index_Drawdown_Percentage'] = np.where(report_df['Index_Peak_Value_Raw'] == 0, 0, report_df['Index_Drawdown_Value'] / report_df['Index_Peak_Value_Raw'])
            index_max_drawdown = report_df['Index_Drawdown_Percentage'].min() if not report_df['Index_Drawdown_Percentage'].empty else 0
            
            print("\n電子業指數績效 (同期比較):")
            print("-" * 30)
            print(f"總報酬率: {index_total_return:.2%}")
            print(f"年化報酬率: {index_annualized_return:.2%}")
            print(f"年化波動率: {index_annualized_volatility:.2%}")
            print(f"夏普比率 (Sharpe Ratio): {index_sharpe_ratio:.2f}")
            print(f"最大回撤 (Max Drawdown): {index_max_drawdown:.2%}")
            print("-" * 30)

            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 9), gridspec_kw={'height_ratios': [3, 1]})
            ax1.plot(report_df.index, report_df['Portfolio_Value'], label='策略淨值', color='blue', linewidth=2)
            ax1.plot(report_df.index, report_df['Index_Value_Plot'], label='電子業指數', color='green', linestyle='--', linewidth=2)
            ax1.set_title('策略資產淨值 vs. 電子業指數')
            ax1.set_ylabel('資產價值 ($)')
            ax1.grid(True)
            ax1.legend(loc='upper left')
            
            ax2.fill_between(report_df.index, report_df['Strategy_Drawdown_Percentage'], 0, color='red', alpha=0.3, label='策略回撤')
            ax2.fill_between(report_df.index, report_df['Index_Drawdown_Percentage'], 0, color='lightgreen', alpha=0.3, label='指數回撤')
            ax2.set_title('回撤圖 (百分比)')
            ax2.set_ylabel('回撤比例')
            ax2.set_xlabel('日期')
            ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)) 
            ax2.grid(True)
            ax2.legend(loc='lower left')
            
            plt.tight_layout()
            plt.show()

        else:
            print("\n警告: 電子業指數數據處理後為空或日期不匹配，無法進行比較。")
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            ax1.plot(report_df.index, report_df['Portfolio_Value'], label='策略淨值', color='blue')
            ax1.set_title('策略資產淨值曲線'); ax1.set_ylabel('資產價值 ($)'); ax1.grid(True); ax1.legend()
            ax2.fill_between(report_df.index, report_df['Strategy_Drawdown_Percentage'], 0, color='red', alpha=0.3, label='策略回撤')
            ax2.set_title('回撤圖 (百分比)'); ax2.set_ylabel('回撤比例'); ax2.set_xlabel('日期'); 
            ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)); ax2.grid(True); ax2.legend()
            plt.tight_layout(); plt.show()

    except FileNotFoundError:
        print(f"\n錯誤: 指數檔案 '{cfg.INDEX_FILE}' 未找到，無法進行指數比較。")
    except Exception as e:
        print(f"\n處理指數數據時發生錯誤: {e}")

    excel_report_filename = f"strategy_performance_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    try:
        columns_to_export = ['Portfolio_Value', 'Strategy_Daily_Return', 'Strategy_Cumulative_Return_Factor', 
                             'Strategy_Drawdown_Percentage']
        if 'Index_Value_Plot' in report_df.columns: 
            columns_to_export.extend(['Index_Value_Plot', 'Index_Daily_Return', 'Index_Cumulative_Return_Factor', 
                                      'Index_Drawdown_Percentage'])
        
        report_df_to_export = report_df[columns_to_export]
        #report_df_to_export.to_excel(excel_report_filename)
        print(f"\n詳細績效報告已匯出至: {excel_report_filename}")
    except Exception as e:
        print(f"\n匯出績效報告至Excel失敗: {e}")

print("\n程式執行完畢。")
