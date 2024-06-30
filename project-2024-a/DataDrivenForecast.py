import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取合并后的数据
final_df = pd.read_csv('project-2024-a/data/final_output.csv')

# 确保日期列为 datetime 类型
final_df['fhjl_time'] = pd.to_datetime(final_df['fhjl_time'])

# 检查列名
print("数据框的列名：")
print(final_df.columns)

# 过滤2023年的数据
data_2023 = final_df[(final_df['fhjl_time'] >= '2023-01-01') & (final_df['fhjl_time'] < '2024-01-01')]

# 检查 ywlx_code 列的唯一值
print("ywlx_code 列的唯一值：")
print(data_2023['ywlx_code'].unique())

# 确保 'ywlx_code' 列存在
if 'ywlx_code' not in data_2023.columns:
    print("Error: 'ywlx_code' 列不存在于数据框中。")
else:
    # 使用字符串匹配来过滤包含“配送”或“自提”的记录
    delivery_methods = data_2023[data_2023['ywlx_code'].str.contains('配送|自提', na=False)]

    # 分析自提与配送的变化
    delivery_stats = delivery_methods.groupby([delivery_methods['fhjl_time'].dt.to_period('M'), 'ywlx_code']).agg({'fhdw': 'sum', 'hk': 'sum'}).reset_index()
    delivery_stats['fhjl_time'] = delivery_stats['fhjl_time'].dt.to_timestamp()

    print("2023年月度自提与配送销量和销售额：")
    print(delivery_stats)

    # 绘制自提与配送的变化趋势图
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=delivery_stats, x='fhjl_time', y='fhdw', hue='ywlx_code', marker='o')
    plt.title('2023年月度自提与配送销量（吨）')
    plt.xlabel('月份')
    plt.ylabel('销量（吨）')
    plt.legend(title='配送方式')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=delivery_stats, x='fhjl_time', y='hk', hue='ywlx_code', marker='o')
    plt.title('2023年月度自提与配送销售额（元）')
    plt.xlabel('月份')
    plt.ylabel('销售额（元）')
    plt.legend(title='配送方式')
    plt.grid(True)
    plt.show()

# 总销量（吨）和总销售额（元）
total_volume = data_2023.groupby('hplx')['fhdw'].sum()
total_revenue = data_2023.groupby('hplx')['hk'].sum()

print("2023年总销量（吨）：")
print(total_volume)
print("\n2023年总销售额（元）：")
print(total_revenue)

# 按月统计
monthly_stats = data_2023.groupby([data_2023['fhjl_time'].dt.to_period('M'), 'hplx']).agg({'fhdw': 'sum', 'hk': 'sum'}).reset_index()
monthly_stats['fhjl_time'] = monthly_stats['fhjl_time'].dt.to_timestamp()

print("2023年月度销量和销售额：")
print(monthly_stats)

# 绘制月度销量和销售额的折线图
plt.figure(figsize=(14, 7))
sns.lineplot(data=monthly_stats, x='fhjl_time', y='fhdw', hue='hplx', marker='o')
plt.title('2023年月度销量（吨）')
plt.xlabel('月份')
plt.ylabel('销量（吨）')
plt.legend(title='货品类型')
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
sns.lineplot(data=monthly_stats, x='fhjl_time', y='hk', hue='hplx', marker='o')
plt.title('2023年月度销售额（元）')
plt.xlabel('月份')
plt.ylabel('销售额（元）')
plt.legend(title='货品类型')
plt.grid(True)
plt.show()

# 平均单价
average_price = data_2023.groupby('hplx')['dj'].mean()

print("2023年平均单价（元/吨）：")
print(average_price)

# 客户总需求量（吨）和总销售额（元）
customer_stats = data_2023.groupby('khmc').agg({'fhdw': 'sum', 'hk': 'sum', 'fhjl_id': 'count'}).rename(columns={'fhjl_id': 'order_count'}).reset_index()

# 计算客户贡献度
total_revenue_sum = customer_stats['hk'].sum()
customer_stats['contribution'] = customer_stats['hk'] / total_revenue_sum * 100

print("2023年客户需求量分析：")
print(customer_stats)

# 过滤掉负值
customer_stats = customer_stats[customer_stats['fhdw'] > 0]
customer_stats = customer_stats[customer_stats['hk'] > 0]

# 选择前15个客户
top_15_customers = customer_stats.nlargest(15, 'fhdw')

# 绘制客户需求量和销售额的饼状图
plt.figure(figsize=(10, 7))
top_15_customers.set_index('khmc')['fhdw'].plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('2023年客户需求量（吨）前15名')
plt.ylabel('')
plt.show()

plt.figure(figsize=(10, 7))
top_15_customers.set_index('khmc')['hk'].plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('2023年客户销售额（元）前15名')
plt.ylabel('')
plt.show()

# 销售经理总销量（吨）和总销售额（元）
sales_manager_stats = data_2023.groupby('sales_name').agg({'fhdw': 'sum', 'hk': 'sum', 'fhjl_id': 'count'}).rename(columns={'fhjl_id': 'order_count'}).reset_index()

# 计算销售经理贡献度
total_revenue_sum = sales_manager_stats['hk'].sum()
sales_manager_stats['contribution'] = sales_manager_stats['hk'] / total_revenue_sum * 100

print("2023年销售经理贡献分析：")
print(sales_manager_stats)

# 过滤掉负值
sales_manager_stats = sales_manager_stats[sales_manager_stats['fhdw'] > 0]
sales_manager_stats = sales_manager_stats[sales_manager_stats['hk'] > 0]

# 绘制销售经理销量和销售额的饼状图
plt.figure(figsize=(10, 7))
sales_manager_stats.set_index('sales_name')['fhdw'].plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('2023年销售经理销量（吨）')
plt.ylabel('')
plt.show()

plt.figure(figsize=(10, 7))
sales_manager_stats.set_index('sales_name')['hk'].plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('2023年销售经理销售额（元）')
plt.ylabel('')
plt.show()

# 准备时间序列数据（使用所有数据进行预测）
monthly_stats_all = final_df.groupby([final_df['fhjl_time'].dt.to_period('M'), 'hplx']).agg({'fhdw': 'sum', 'hk': 'sum'}).reset_index()
monthly_stats_all['fhjl_time'] = monthly_stats_all['fhjl_time'].dt.to_timestamp()

monthly_volume_all = monthly_stats_all.pivot(index='fhjl_time', columns='hplx', values='fhdw').fillna(0)
monthly_revenue_all = monthly_stats_all.pivot(index='fhjl_time', columns='hplx', values='hk').fillna(0)

# 准备特征和目标变量
X = np.array((monthly_volume_all.index - monthly_volume_all.index[0]).days).reshape(-1, 1)  # 将日期转换为天数
y_volume = monthly_volume_all.values
y_revenue = monthly_revenue_all.values

# 拆分训练集和测试集
X_train, X_test, y_volume_train, y_volume_test = train_test_split(X, y_volume, test_size=0.2, random_state=42)
X_train, X_test, y_revenue_train, y_revenue_test = train_test_split(X, y_revenue, test_size=0.2, random_state=42)

# 建立线性回归模型
model_volume = LinearRegression()
model_revenue = LinearRegression()

# 训练模型
model_volume.fit(X_train, y_volume_train)
model_revenue.fit(X_train, y_revenue_train)

# 预测
volume_forecast = model_volume.predict(X_test)
revenue_forecast = model_revenue.predict(X_test)

# 确保预测值不为负
volume_forecast = np.maximum(volume_forecast, 0)
revenue_forecast = np.maximum(revenue_forecast, 0)

# 评估模型
volume_mse = mean_squared_error(y_volume_test, volume_forecast)
volume_r2 = r2_score(y_volume_test, volume_forecast)
revenue_mse = mean_squared_error(y_revenue_test, revenue_forecast)
revenue_r2 = r2_score(y_revenue_test, revenue_forecast)

print(f"销量预测 MSE: {volume_mse}, R2: {volume_r2}")
print(f"销售额预测 MSE: {revenue_mse}, R2: {revenue_r2}")

# 预测未来数据到2025年
future_dates = pd.date_range(start=monthly_volume_all.index[-1], periods=25, freq='ME')
future_X = np.array((future_dates - monthly_volume_all.index[0]).days).reshape(-1, 1)

future_volume_forecast = model_volume.predict(future_X)
future_revenue_forecast = model_revenue.predict(future_X)

# 确保未来预测值不为负
future_volume_forecast = np.maximum(future_volume_forecast, 0)
future_revenue_forecast = np.maximum(future_revenue_forecast, 0)

# 合并实际和预测数据
all_dates = monthly_volume_all.index.append(future_dates)
all_volume = np.vstack([monthly_volume_all.values, future_volume_forecast])
all_revenue = np.vstack([monthly_revenue_all.values, future_revenue_forecast])

# 绘制预测的折线图
plt.figure(figsize=(14, 7))
for i, col in enumerate(monthly_volume_all.columns):
    plt.plot(all_dates[:len(monthly_volume_all)], monthly_volume_all[col], label=f'{col} 实际')
    plt.plot(all_dates[len(monthly_volume_all):], future_volume_forecast[:, i], '--', label=f'{col} 预测')
plt.title('月度销量预测（吨）')
plt.xlabel('月份')
plt.ylabel('销量（吨）')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
for i, col in enumerate(monthly_revenue_all.columns):
    plt.plot(all_dates[:len(monthly_revenue_all)], monthly_revenue_all[col], label=f'{col} 实际')
    plt.plot(all_dates[len(monthly_volume_all):], future_revenue_forecast[:, i], '--', label=f'{col} 预测')
plt.title('月度销售额预测（元）')
plt.xlabel('月份')
plt.ylabel('销售额（元）')
plt.legend()
plt.grid(True)
plt.show()

# 预测年度总销量和总销售额
annual_volume_forecast = np.sum(future_volume_forecast, axis=0)
annual_revenue_forecast = np.sum(future_revenue_forecast, axis=0)

print("\n年度总销量预测（吨）：")
print(annual_volume_forecast)
print("\n年度总销售额预测（元）：")
print(annual_revenue_forecast)

# 计算去年总销量
last_year_total_volume = monthly_volume_all.sum()

# 预测今年总销量
this_year_total_volume = annual_volume_forecast

# 计算增长率
growth_rate_volume = (this_year_total_volume - last_year_total_volume) / last_year_total_volume * 100

print("\n销量增长率预测：")
print(growth_rate_volume)

# 分析前15位客人的常用配送方式
top_15_customers_delivery = data_2023[data_2023['khmc'].isin(top_15_customers['khmc'])]

customer_delivery_stats = top_15_customers_delivery[top_15_customers_delivery['ywlx_code'].str.contains('配送|自提', na=False)].groupby(['khmc', 'ywlx_code']).agg({'fhdw': 'sum', 'hk': 'sum', 'fhjl_id': 'count'}).rename(columns={'fhjl_id': 'order_count'}).reset_index()

print("前15位客户的常用配送方式：")
print(customer_delivery_stats)