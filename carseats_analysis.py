# carseats_analysis.py

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

# ==========================================
# 第1步：准备数据（为了防止报错，加了备用数据方案）
# ==========================================
try:
    from pydataset import data
    carseats = data('Carseats')
    print("成功加载真实的 Carseats 数据集！")
except:
    print("没有pydataset库，生成一份模拟数据供代码跑通...")
    np.random.seed(42)
    carseats = pd.DataFrame({
        'Sales': np.random.normal(7.5, 2.5, 400),       # 销售额
        'Price': np.random.uniform(20, 200, 400),       # 价格
        'Income': np.random.normal(68, 28, 400),        # 收入
        'Advertising': np.random.uniform(0, 30, 400),   # 广告费
        'ShelveLoc': np.random.choice(['Bad', 'Medium', 'Good'], 400) # 货架位置
    })

# ==========================================
# 第2步：把文字（定性特征）变成数字
# ==========================================
# ShelveLoc 是文本（Bad/Medium/Good），模型不认识，需要用 get_dummies 变成0和1
# drop_first=True 的意思是：把第一个（Bad）作为“基准组”，只给 Medium 和 Good 生成列
carseats_encoded = pd.get_dummies(carseats, columns=['ShelveLoc'], drop_first=True, dtype=int)

# 定义 X（自变量）和 y（因变量）
X = carseats_encoded[['Price', 'Income', 'Advertising', 'ShelveLoc_Good', 'ShelveLoc_Medium']]
y = carseats_encoded['Sales']

# 给 X 加上常数项（截距），这是做回归的固定套路
X = sm.add_constant(X)

# ==========================================
# 第3步：建立模型并打印结果
# ==========================================
model = sm.OLS(y, X).fit()
print("\n========== 模型拟合报告 ==========")
print(model.summary())

# ==========================================
# 第4步：回答问题
# ==========================================
print("\n========== 问题解答 ==========")

# 问题1：基准组是谁？
print("1. ShelveLoc 的基准组是：'Bad'。")
print("   （因为在生成虚拟变量时，Bad 被丢掉了，它是用来做对比的参照物。）")

# 问题2：ShelveLoc[Good] 系数的商业含义？
coef_good = model.params.get('ShelveLoc_Good', 0)
print(f"\n2. ShelveLoc_Good 的系数是：{coef_good:.4f}")
print(f"   商业含义：在其他条件（价格、收入、广告）都不变的情况下，")
print(f"   把货架位置从 'Bad' 换成 'Good'，销售额平均会增加 {coef_good:.4f}。")

# 问题3：计算 VIF 评估多重共线性
print("\n3. 计算各变量的 VIF（评估多重共线性）：")
X_vif = X.copy()
vif_data = pd.DataFrame()
vif_data["变量名"] = X_vif.columns
vif_data["VIF值"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print(vif_data)

# 给出简单结论
max_vif = vif_data['VIF值'].max()
if max_vif > 10:
    print("结论：VIF最大值大于10，存在严重的多重共线性风险！")
elif max_vif > 5:
    print("结论：VIF最大值大于5，存在一定的多重共线性风险，需要留意。")
else:
    print("结论：所有变量的VIF都小于5，不存在多重共线性风险，模型很健康。")