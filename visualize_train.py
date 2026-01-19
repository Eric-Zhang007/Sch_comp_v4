import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
sns.set_style("whitegrid")

# ==========================================
# 1. 读取结果
# ==========================================
# 请修改这里的路径为你实际的文件路径
file_path = "./predictions.csv" 

try:
    # 指定 index_col="Time" 以便正确解析时间索引
    df = pd.read_csv(file_path, index_col="Time", parse_dates=True)
    print(f"成功读取数据，形状: {df.shape}")
except Exception as e:
    print(f"读取文件失败: {e}")
    exit()

# ==========================================
# 2. 画图诊断
# ==========================================
plt.figure(figsize=(16, 12))

# --- 子图 1: 整体预测 vs 真实值 ---
plt.subplot(3, 1, 1)
plt.plot(df.index, df['y'], label='Actual (y)', color='black', alpha=0.6, linewidth=1)
plt.plot(df.index, df['y_pred'], label='Predicted (y_pred)', color='#d62728', alpha=0.7, linewidth=1)
plt.title("Total Prediction vs Actual", fontsize=14)
plt.ylabel("Load Value")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# --- 子图 2: Prophet 趋势项诊断 ---
# 检查 Prophet 提取的趋势(Trend)是否捕捉到了整体走势
plt.subplot(3, 1, 2)
plt.plot(df.index, df['y'], label='Actual (y)', color='gray', alpha=0.3)
plt.plot(df.index, df['trend'], label='Prophet Trend', color='#1f77b4', linewidth=1.5)
plt.title("Prophet Trend Component vs Actual", fontsize=14)
plt.ylabel("Trend Value")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# --- 子图 3: LSTM 残差预测诊断 ---
# 检查 LSTM 是否成功预测了残差 (resid)
plt.subplot(3, 1, 3)
plt.plot(df.index, df['resid'], label='True Residual (resid)', color='gray', alpha=0.5)
plt.plot(df.index, df['resid_pred'], label='Predicted Residual (resid_pred)', color='#2ca02c', alpha=0.8, linewidth=1)
plt.title("LSTM Residual Prediction (High Frequency Component)", fontsize=14)
plt.ylabel("Residual Value")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==========================================
# 3. 简单的统计输出
# ==========================================
print("\n--- 简单统计分析 ---")
print(df[['y', 'y_pred']].describe())

# 检查是否有极端的预测偏差
df['error'] = df['y'] - df['y_pred']
max_err = df['error'].abs().max()
print(f"\n最大绝对误差: {max_err:.4f}")

# 检查残差预测的方差，如果 resid_pred 方差极小，说明 LSTM 没学到东西（发生了坍塌）
resid_std = df['resid'].std()
resid_pred_std = df['resid_pred'].std()
print(f"真实残差标准差: {resid_std:.4f}")
print(f"预测残差标准差: {resid_pred_std:.4f}")
print(f"恢复率 (Pred Std / True Std): {resid_pred_std/resid_std:.2%}")
if resid_pred_std / resid_std < 0.2:
    print("警告: 预测残差的波动幅度远小于真实残差，模型可能发生‘趋中’(Collapse)，建议增加特征或调整 mask 策略。")