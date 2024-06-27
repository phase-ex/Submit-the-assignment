import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# 读取生成的中间表
final_df = pd.read_csv('project-2024-a/data/final_output.csv')

# 1. 去年一年水泥和矿粉各自的销量情况
last_year = final_df[final_df['fhjl_time'].str.contains('2022')]
cement_sales = last_year[last_year['hplx'] == '水泥']['fhdw'].sum()
mineral_powder_sales = last_year[last_year['hplx'] == '矿粉']['fhdw'].sum()

print(f"去年水泥销量: {cement_sales} 吨")
print(f"去年矿粉销量: {mineral_powder_sales} 吨")

# 2. 去年客户需求量分析，哪些客户最重要
customer_demand = last_year.groupby('khmc')['fhdw'].sum().sort_values(ascending=False)
print("去年客户需求量分析:")
print(customer_demand)

# 3. 去年哪些销售经理贡献最大
sales_contribution = last_year.groupby('sales_name')['fhdw'].sum().sort_values(ascending=False)
print("去年销售经理贡献分析:")
print(sales_contribution)

# 4. 预测今年的水泥和矿粉的销量
# 准备数据
final_df['year'] = pd.to_datetime(final_df['fhjl_time']).dt.year

# 水泥销量预测
cement_data = final_df[final_df['hplx'] == '水泥'].groupby('year')['fhdw'].sum().reset_index()
X_cement = cement_data[['year']]
y_cement = cement_data['fhdw']
model_cement = LinearRegression()
model_cement.fit(X_cement, y_cement)

# 预测2023年的水泥销量
next_year = np.array([[2023]])
predicted_cement_sales = model_cement.predict(next_year)
print(f"预测2023年水泥销量: {predicted_cement_sales[0]} 吨")

# 矿粉销量预测
mineral_powder_data = final_df[final_df['hplx'] == '矿粉'].groupby('year')['fhdw'].sum().reset_index()
X_mineral_powder = mineral_powder_data[['year']]
y_mineral_powder = mineral_powder_data['fhdw']
model_mineral_powder = LinearRegression()
model_mineral_powder.fit(X_mineral_powder, y_mineral_powder)

# 预测2023年的矿粉销量111

predicted_mineral_powder_sales = model_mineral_powder.predict(next_year)
print(f"预测2023年矿粉销量: {predicted_mineral_powder_sales[0]} 吨")