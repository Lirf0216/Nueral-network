"""
课堂作业1：探究糖尿病预测的关键特征
探究糖尿病数据集中哪些特征对预测糖尿病进展最重要

参考方法：
1. 使用所有10个特征进行线性回归，分析特征权重（系数）
2. 对每个特征分别建立独立的线性回归模型，比较均方误差（MSE）
"""

import numpy as np  # 导入numpy库，用于数值计算
import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘图
from sklearn import datasets, linear_model, model_selection  # 导入sklearn库中的数据集、线性回归模型和模型选择模块
from sklearn.metrics import mean_squared_error  # 导入均方误差评估指标
from sklearn.preprocessing import StandardScaler  # 导入标准化处理器

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 加载糖尿病数据集
print("=" * 60)  # 打印分隔线
print("作业1：糖尿病预测关键特征分析")  # 打印标题
print("=" * 60)  # 打印分隔线

X, y = datasets.load_diabetes(return_X_y=True)  # 加载糖尿病数据集，返回特征X和标签y
print(f"数据集形状: X={X.shape}, y={y.shape}")  # 打印数据集形状
print(f"特征数量: {X.shape[1]}")  # 打印特征数量
print(f"样本数量: {X.shape[0]}")  # 打印样本数量

# 显示特征名称
diabetes = datasets.load_diabetes()  # 加载完整的糖尿病数据集对象
feature_names = diabetes.feature_names  # 获取特征名称
print(f"\n特征名称: {list(feature_names)}")  # 打印特征名称列表
print("对应索引: age(0), sex(1), bmi(2), bp(3), s1(4), s2(5), s3(6), s4(7), s5(8), s6(9)")  # 打印特征索引对应关系

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.2, random_state=42  # 将数据集按80%训练集、20%测试集划分，设置随机种子为42
)

# 标准化特征（对线性回归很重要）
scaler = StandardScaler()  # 创建标准化处理器对象
X_train_scaled = scaler.fit_transform(X_train)  # 对训练集特征进行拟合和转换
X_test_scaled = scaler.transform(X_test)  # 对测试集特征进行转换（使用训练集的参数）

# ============================
# 方法1: 多特征线性回归（分析权重）
# ============================
print("\n" + "=" * 60)  # 打印分隔线
print("方法1: 多特征线性回归 - 分析特征权重")  # 打印方法标题
print("=" * 60)  # 打印分隔线

# 构建并训练多特征线性回归模型
multi_model = linear_model.LinearRegression()  # 创建线性回归模型对象
multi_model.fit(X_train_scaled, y_train)  # 使用训练数据拟合模型

# 获取模型系数（权重）
coefficients = multi_model.coef_  # 获取模型系数（特征权重）
intercept = multi_model.intercept_  # 获取模型截距

# 创建特征权重字典
feature_weights = dict(zip(feature_names, coefficients))  # 将特征名称和系数组合成字典
print(f"模型截距: {intercept:.4f}")  # 打印模型截距

# 按权重绝对值降序排序
print("\n特征权重排序（按绝对值降序）:")  # 打印标题
print("-" * 50)  # 打印分隔线
sorted_weights = sorted(feature_weights.items(), key=lambda x: abs(x[1]), reverse=True)  # 按权重绝对值降序排序

for i, (feature, weight) in enumerate(sorted_weights, 1):  # 遍历排序后的特征权重
    print(
        f"{i:2d}. {feature:15s}: {weight:8.4f} {'(正相关)' if weight > 0 else '(负相关)'}"  # 打印特征权重信息
    )

# 可视化特征权重
plt.figure(figsize=(10, 6))  # 创建图形，设置大小为10x6英寸
features = [item[0] for item in sorted_weights]  # 获取排序后的特征名称列表
weights = [item[1] for item in sorted_weights]  # 获取排序后的权重值列表

colors = ["green" if w > 0 else "red" for w in weights]  # 根据权重正负设置颜色：正相关为绿色，负相关为红色
bars = plt.barh(features, weights, color=colors)  # 创建水平条形图
plt.xlabel("权重系数", fontsize=12)  # 设置x轴标签
plt.title("糖尿病预测特征重要性（基于多特征线性回归权重）", fontsize=14)  # 设置图形标题
plt.axvline(x=0, color="black", linestyle="-", linewidth=0.5)  # 在x=0处添加垂直参考线

# 在条形上添加数值标签
for bar, weight in zip(bars, weights):  # 遍历每个条形和对应的权重值
    width = bar.get_width()  # 获取条形的宽度
    label_x = width + (0.01 if width >= 0 else -0.01)  # 计算标签的x坐标位置
    plt.text(  # 添加文本标签
        label_x,
        bar.get_y() + bar.get_height() / 2,  # 计算标签的y坐标位置（条形中心）
        f"{weight:.3f}",  # 设置标签文本内容（保留3位小数）
        ha="left" if width >= 0 else "right",  # 设置水平对齐方式
        va="center",  # 设置垂直对齐方式
    )

plt.tight_layout()  # 自动调整子图参数
plt.show()  # 显示图形

# ============================
# 方法2: 单特征线性回归（比较MSE）
# ============================
print("\n" + "=" * 60)  # 打印分隔线
print("方法2: 单特征线性回归 - 比较均方误差(MSE)")  # 打印方法标题
print("=" * 60)  # 打印分隔线

mse_results = {}  # 创建空字典用于存储MSE结果
feature_indices = range(X.shape[1])  # 获取特征索引范围

print("各特征单独建模的预测效果:")  # 打印标题
print("-" * 70)  # 打印分隔线
print(f"{'特征':15s} {'特征名':15s} {'训练集MSE':12s} {'测试集MSE':12s} {'排名':5s}")  # 打印表头
print("-" * 70)  # 打印分隔线

for idx in feature_indices:  # 遍历每个特征索引
    # 提取单个特征
    X_train_single = X_train_scaled[:, idx].reshape(-1, 1)  # 提取训练集单个特征并重塑为二维数组
    X_test_single = X_test_scaled[:, idx].reshape(-1, 1)  # 提取测试集单个特征并重塑为二维数组

    # 构建并训练单特征线性回归模型
    single_model = linear_model.LinearRegression()  # 创建线性回归模型对象
    single_model.fit(X_train_single, y_train)  # 使用单个特征训练数据拟合模型

    # 预测
    y_train_pred = single_model.predict(X_train_single)  # 使用训练集进行预测
    y_test_pred = single_model.predict(X_test_single)  # 使用测试集进行预测

    # 计算MSE
    train_mse = mean_squared_error(y_train, y_train_pred)  # 计算训练集均方误差
    test_mse = mean_squared_error(y_test, y_test_pred)  # 计算测试集均方误差

    mse_results[feature_names[idx]] = {  # 将结果存储到字典中
        "train_mse": train_mse,  # 训练集MSE
        "test_mse": test_mse,  # 测试集MSE
        "index": idx,  # 特征索引
    }

# 按测试集MSE升序排序（MSE越小越好）
sorted_mse = sorted(mse_results.items(), key=lambda x: x[1]["test_mse"])  # 按测试集MSE升序排序

for rank, (feature, results) in enumerate(sorted_mse, 1):  # 遍历排序后的MSE结果
    print(  # 打印每个特征的MSE结果
        f"{feature:15s} {feature_names[results['index']]:15s} "
        f"{results['train_mse']:12.2f} {results['test_mse']:12.2f} {rank:5d}"
    )

# 可视化MSE结果
plt.figure(figsize=(10, 6))  # 创建图形，设置大小为10x6英寸
features_mse = [item[0] for item in sorted_mse]  # 获取排序后的特征名称列表
test_mses = [item[1]["test_mse"] for item in sorted_mse]  # 获取排序后的测试集MSE值列表

bars = plt.barh(features_mse, test_mses, color="skyblue")  # 创建水平条形图，颜色为天蓝色
plt.xlabel("测试集均方误差(MSE)", fontsize=12)  # 设置x轴标签
plt.title("糖尿病预测特征效果比较（基于单特征模型MSE）", fontsize=14)  # 设置图形标题

# 在条形上添加MSE数值
for bar, mse in zip(bars, test_mses):  # 遍历每个条形和对应的MSE值
    width = bar.get_width()  # 获取条形的宽度
    plt.text(  # 添加文本标签
        width + 50,  # 计算标签的x坐标位置（条形右侧偏移50）
        bar.get_y() + bar.get_height() / 2,  # 计算标签的y坐标位置（条形中心）
        f"{mse:.1f}",  # 设置标签文本内容（保留1位小数）
        ha="left",  # 设置水平对齐方式为左对齐
        va="center",  # 设置垂直对齐方式为居中
    )

plt.tight_layout()  # 自动调整子图参数
plt.show()  # 显示图形

# ============================
# 综合分析
# ============================
print("\n" + "=" * 60)  # 打印分隔线
print("综合分析结果")  # 打印标题
print("=" * 60)  # 打印分隔线

print("\n根据方法1（特征权重绝对值）的前3重要特征:")  # 打印方法1的前3重要特征
for i in range(3):  # 遍历前3个特征
    feature, weight = sorted_weights[i]  # 获取特征名称和权重
    print(f"  {i+1}. {feature}: 权重={weight:.4f}")  # 打印特征信息

print("\n根据方法2（测试集MSE）的前3有效特征:")  # 打印方法2的前3有效特征
for i in range(3):  # 遍历前3个特征
    feature, results = sorted_mse[i]  # 获取特征名称和结果
    print(f"  {i+1}. {feature}: MSE={results['test_mse']:.2f}")  # 打印特征信息

print("\n" + "=" * 60)  # 打印分隔线
print("结论:")  # 打印结论标题
print("-" * 60)  # 打印分隔线
print("1. 两种方法都显示 BMI(bmi) 和 s5 是最重要的特征")  # 打印结论1
print("2. 方法1通过权重分析显示了特征的正负相关性")  # 打印结论2
print("3. 方法2通过单特征MSE评估了每个特征的独立预测能力")  # 打印结论3
print("4. 综合来看，BMI是预测糖尿病进展的最关键特征")  # 打印结论4
print("=" * 60)  # 打印分隔线

# 3. 最终的多特征模型评估
print(f"\n最终多特征模型性能评估:")  # 打印评估标题
y_pred_multi = multi_model.predict(X_test_scaled)  # 使用多特征模型对测试集进行预测
final_mse = mean_squared_error(y_test, y_pred_multi)  # 计算最终均方误差
final_r2 = multi_model.score(X_test_scaled, y_test)  # 计算R²分数

print(f"测试集均方误差(MSE): {final_mse:.2f}")  # 打印MSE
print(f"测试集R²分数: {final_r2:.4f}")  # 打印R²分数