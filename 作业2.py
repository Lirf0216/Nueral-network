# homework2_logistic_regression.py
"""
课堂作业2：使用逻辑回归模型进行分类任务

作业要求：
1. 从 sklearn.datasets 中加载一个分类数据集（鸢尾花数据集 load_iris）
2. 打印数据集的样本总数
3. 选择并打印任意一个样本的特征及其对应的类别标签
4. 统计并打印数据集中每个类别的样本数量，了解其分布情况
5. 模型训练，包括划分训练集和测试集，使用逻辑回归模型在训练集进行训练
6. 打印训练好的模型的参数，即模型的系数和截距
7. 使用训练好的模型对测试集进行预测，计算并打印模型在测试集上的准确率
8. 选取测试集中的前5个样本，打印这5个样本的预测标签和它们的真实标签
"""

import numpy as np  # 导入numpy库，用于数值计算
from sklearn.datasets import load_iris  # 导入鸢尾花数据集
from sklearn.model_selection import train_test_split  # 导入数据集划分函数
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归模型
from sklearn.metrics import accuracy_score, classification_report  # 导入评估指标
import pandas as pd  # 导入pandas库，用于数据处理

# ============================
# 1. 加载鸢尾花数据集
# ============================
print("=" * 70)  # 打印分隔线
print("作业2：鸢尾花分类 - 逻辑回归模型")  # 打印作业标题
print("=" * 70)  # 打印分隔线

iris = load_iris()  # 加载鸢尾花数据集
X = iris.data  # 获取特征数据
y = iris.target  # 获取标签数据
feature_names = iris.feature_names  # 获取特征名称
target_names = iris.target_names  # 获取类别名称

# ============================
# 2. 打印数据集的样本总数
# ============================
print(f"\n1. 数据集基本信息:")  # 打印基本信息标题
print(f"   样本总数: {X.shape[0]} 个")  # 打印样本总数
print(f"   特征数量: {X.shape[1]} 个")  # 打印特征数量
print(f"   特征名称: {', '.join(feature_names)}")  # 打印所有特征名称
print(f"   类别名称: {', '.join(target_names)}")  # 打印所有类别名称
print(f"   类别编码: 0={target_names[0]}, 1={target_names[1]}, 2={target_names[2]}")  # 打印类别编码对应关系

# ============================
# 3. 选择并打印任意一个样本的特征和标签
# ============================
print(f"\n2. 随机选择一个样本进行展示:")  # 打印样本展示标题
sample_idx = 42  # 选择第43个样本（索引从0开始）
sample_features = X[sample_idx]  # 获取样本特征
sample_label = y[sample_idx]  # 获取样本标签
sample_label_name = target_names[sample_label]  # 获取标签对应的类别名称

print(f"   样本索引: {sample_idx}")  # 打印样本索引
print(f"   特征值:")  # 打印特征值标题
for i, (feature_name, value) in enumerate(zip(feature_names, sample_features)):  # 遍历特征和值
    print(f"     {feature_name:20s}: {value:.4f}")  # 打印每个特征的值
print(f"   类别标签: {sample_label} ({sample_label_name})")  # 打印类别标签

# ============================
# 4. 统计并打印每个类别的样本数量
# ============================
print(f"\n3. 数据集类别分布情况:")  # 打印类别分布标题
unique, counts = np.unique(y, return_counts=True)  # 统计每个类别的数量
total_samples = len(y)  # 计算总样本数

for label, count in zip(unique, counts):  # 遍历每个类别
    percentage = (count / total_samples) * 100  # 计算百分比
    print(
        f"   类别 {label} ({target_names[label]}): {count:3d} 个样本 ({percentage:.1f}%)"  # 打印类别信息
    )

print(f"   总计: {total_samples} 个样本")  # 打印总样本数

# 可视化类别分布（可选）
import matplotlib.pyplot as plt  # 导入matplotlib绘图库

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

plt.figure(figsize=(8, 5))  # 创建图形，设置大小
bars = plt.bar(target_names, counts, color=["lightcoral", "lightgreen", "lightblue"])  # 创建条形图
plt.title("鸢尾花数据集类别分布", fontsize=14)  # 设置标题
plt.xlabel("类别", fontsize=12)  # 设置x轴标签
plt.ylabel("样本数量", fontsize=12)  # 设置y轴标签

# 在条形上添加数量标签
for bar, count in zip(bars, counts):  # 遍历每个条形
    height = bar.get_height()  # 获取条形高度
    plt.text(  # 添加文本标签
        bar.get_x() + bar.get_width() / 2.0,  # 计算x坐标（条形中心）
        height + 0.5,  # 计算y坐标（条形上方）
        f"{count}",  # 设置文本内容
        ha="center",  # 水平居中
        va="bottom",  # 垂直底部对齐
    )

plt.tight_layout()  # 自动调整布局
plt.show()  # 显示图形

# ============================
# 5. 划分训练集和测试集，训练逻辑回归模型
# ============================
print(f"\n4. 划分数据集并训练模型:")  # 打印划分数据集标题
X_train, X_test, y_train, y_test = train_test_split(  # 划分数据集
    X, y, test_size=0.3, random_state=42, stratify=y  # 30%测试集，设置随机种子，分层抽样
)

print(f"   训练集大小: {X_train.shape[0]} 个样本")  # 打印训练集大小
print(f"   测试集大小: {X_test.shape[0]} 个样本")  # 打印测试集大小

# 创建并训练逻辑回归模型
print(f"\n   训练逻辑回归模型...")  # 打印训练开始信息
model = LogisticRegression(
    max_iter=200,  # 设置最大迭代次数
    random_state=42
)
model.fit(X_train, y_train)  # 训练模型
print("   模型训练完成！")  # 打印训练完成信息

# ============================
# 6. 打印模型参数（系数和截距）
# ============================
print(f"\n5. 模型参数:")  # 打印模型参数标题
print(f"   系数 (coef_):")  # 打印系数标题
coefficients = model.coef_  # 获取模型系数
for i, (class_name, coef_row) in enumerate(zip(target_names, coefficients)):  # 遍历每个类别的系数
    print(f"\n   类别 {i} ({class_name}) 的系数:")  # 打印类别系数标题
    for j, (feature_name, coef_value) in enumerate(zip(feature_names, coef_row)):  # 遍历每个特征的系数
        print(f"     {feature_name:20s}: {coef_value:8.4f}")  # 打印特征系数

print(f"\n   截距 (intercept_):")  # 打印截距标题
for i, (class_name, intercept_value) in enumerate(zip(target_names, model.intercept_)):  # 遍历每个类别的截距
    print(f"   类别 {i} ({class_name}): {intercept_value:8.4f}")  # 打印类别截距

# ============================
# 7. 使用模型进行预测并计算准确率
# ============================
print(f"\n6. 模型预测与评估:")  # 打印预测评估标题
y_pred = model.predict(X_test)  # 使用模型进行预测
accuracy = accuracy_score(y_test, y_pred)  # 计算准确率
print(f"   测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")  # 打印准确率

# 更详细的分类报告
print(f"\n   分类报告:")  # 打印分类报告标题
report = classification_report(y_test, y_pred, target_names=target_names)  # 生成分类报告
print(report)  # 打印分类报告

# ============================
# 8. 选取测试集前5个样本，打印预测和真实标签
# ============================
print(f"\n7. 测试集前5个样本的预测结果:")  # 打印预测结果标题
print("-" * 70)  # 打印分隔线
print(f"{'样本':6s} {'特征值':30s} {'真实标签':12s} {'预测标签':12s} {'是否正确':10s}")  # 打印表头
print("-" * 70)  # 打印分隔线

for i in range(min(5, len(X_test))):  # 遍历前5个测试样本
    # 获取样本特征
    sample = X_test[i]  # 获取第i个样本特征
    true_label = y_test[i]  # 获取第i个真实标签
    pred_label = y_pred[i]  # 获取第i个预测标签

    # 格式化特征值显示
    features_str = ", ".join([f"{val:.2f}" for val in sample])  # 将特征值格式化为字符串
    if len(features_str) > 30:  # 如果字符串太长
        features_str = features_str[:27] + "..."  # 截断并添加省略号

    true_label_str = f"{true_label} ({target_names[true_label]})"  # 格式化真实标签
    pred_label_str = f"{pred_label} ({target_names[pred_label]})"  # 格式化预测标签
    is_correct = "✓" if true_label == pred_label else "✗"  # 判断是否正确

    print(  # 打印结果
        f"{i+1:6d} {features_str:30s} {true_label_str:12s} {pred_label_str:12s} {is_correct:10s}"
    )

# ============================
# 附加：概率预测展示
# ============================
print(f"\n8. 第1个测试样本的详细概率预测:")  # 打印概率预测标题
sample_idx = 0  # 选择第一个测试样本
sample_proba = model.predict_proba([X_test[sample_idx]])[0]  # 获取预测概率
true_label = y_test[sample_idx]  # 获取真实标签

print(f"   样本特征值: {X_test[sample_idx]}")  # 打印样本特征值
print(f"   真实类别: {true_label} ({target_names[true_label]})")  # 打印真实类别
print(f"\n   属于各个类别的概率:")  # 打印概率标题
for i, (class_name, prob) in enumerate(zip(target_names, sample_proba)):  # 遍历每个类别的概率
    print(f"     {class_name:15s}: {prob:.4f} ({prob*100:.1f}%)")  # 打印类别概率

predicted_class = np.argmax(sample_proba)  # 找到概率最大的类别
print(f"\n   预测类别: {predicted_class} ({target_names[predicted_class]})")  # 打印预测类别
print(f"   是否正确: {'是' if predicted_class == true_label else '否'}")  # 打印是否正确

# ============================
# 可视化预测结果
# ============================
plt.figure(figsize=(12, 5))  # 创建图形，设置大小

# 子图1：特征重要性可视化
plt.subplot(1, 2, 1)  # 创建第一个子图
feature_importance = np.mean(np.abs(model.coef_), axis=0)  # 计算特征重要性（平均权重绝对值）
sorted_idx = np.argsort(feature_importance)  # 对特征重要性排序

plt.barh(np.array(feature_names)[sorted_idx], feature_importance[sorted_idx])  # 创建水平条形图
plt.xlabel("平均权重绝对值", fontsize=12)  # 设置x轴标签
plt.title("特征重要性（逻辑回归）", fontsize=14)  # 设置标题
plt.tight_layout()  # 自动调整布局

# 子图2：混淆矩阵可视化
plt.subplot(1, 2, 2)  # 创建第二个子图
from sklearn.metrics import confusion_matrix  # 导入混淆矩阵函数
import seaborn as sns  # 导入seaborn库，用于美化图形

cm = confusion_matrix(y_test, y_pred)  # 计算混淆矩阵
sns.heatmap(  # 创建热力图
    cm,  # 混淆矩阵数据
    annot=True,  # 显示数值
    fmt="d",  # 整数格式
    cmap="Blues",  # 颜色映射
    xticklabels=target_names,  # x轴刻度标签
    yticklabels=target_names,  # y轴刻度标签
)
plt.title("混淆矩阵", fontsize=14)  # 设置标题
plt.xlabel("预测标签", fontsize=12)  # 设置x轴标签
plt.ylabel("真实标签", fontsize=12)  # 设置y轴标签

plt.tight_layout()  # 自动调整布局
plt.show()  # 显示图形

# ============================
# 总结
# ============================
print("\n" + "=" * 70)  # 打印分隔线
print("作业总结")  # 打印总结标题
print("=" * 70)  # 打印分隔线
print(f"1. 数据集: 鸢尾花数据集，共{len(X)}个样本，{len(feature_names)}个特征")  # 打印数据集信息
print(f"2. 分类任务: 3类别分类 ({', '.join(target_names)})")  # 打印分类任务信息
print(f"3. 模型: 逻辑回归 (Logistic Regression)")  # 打印模型信息
print(f"4. 准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")  # 打印准确率
print(f"5. 最重要的特征: {feature_names[np.argmax(feature_importance)]}")  # 打印最重要特征
print("=" * 70)  # 打印分隔线