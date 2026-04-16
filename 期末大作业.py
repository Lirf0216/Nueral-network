"""
葡萄酒质量分类项目 - 分图绘制版本 (英文标签)
数据集：Wine Reviews
任务：葡萄酒质量分类
模型对比：Logistic Regression vs Random Forest
"""

# 1. 导入必要的库
import pandas as pd  # 数据处理和分析库，提供DataFrame数据结构
import numpy as np  # 科学计算库，提供多维数组和数学函数
import matplotlib.pyplot as plt  # 基础绘图库，用于数据可视化
import seaborn as sns  # 统计可视化库，基于matplotlib，更美观
from sklearn.model_selection import train_test_split, GridSearchCV  # 划分数据集和网格搜索超参数
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder  # 标准化、标签编码、独热编码
from sklearn.linear_model import LogisticRegression  # 逻辑回归分类器
from sklearn.ensemble import RandomForestClassifier  # 随机森林分类器（集成学习）
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # 评估指标：准确率、分类报告、混淆矩阵、F1分数
from sklearn.compose import ColumnTransformer  # 对不同列应用不同预处理
from sklearn.pipeline import Pipeline  # 将多个处理步骤打包成管道
import warnings  # 警告控制模块
warnings.filterwarnings('ignore')  # 忽略所有警告信息
import joblib  # 保存和加载机器学习模型
import time  # 时间模块，用于计时和性能监控

# 设置图表样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# 2. 创建模拟数据
def create_simulated_wine_data(n_samples=3000):
    """Create simulated wine dataset"""
    np.random.seed(42)

    print("Creating simulated wine dataset...")

    countries = ['France', 'Italy', 'Spain', 'USA', 'Chile']
    varieties = ['Cabernet', 'Chardonnay', 'Pinot Noir', 'Merlot', 'Syrah']

    data = {
        'country': np.random.choice(countries, n_samples),
        'points': np.random.randint(80, 100, n_samples),
        'price': np.random.exponential(50, n_samples).clip(10, 300).round(2),
        'variety': np.random.choice(varieties, n_samples)
    }

    df = pd.DataFrame(data)

    # Create quality labels
    conditions = [
        (df['points'] >= 92) | ((df['points'] >= 88) & (df['price'] > 100)),
        (df['points'] >= 85) | ((df['points'] >= 82) & (df['price'] > 50)),
        (df['points'] < 82)
    ]

    quality_levels = ['Excellent', 'Good', 'Average']
    df['quality'] = np.select(conditions, quality_levels, default='Good')

    print(f"Dataset created, shape: {df.shape}")
    print(f"Quality distribution:\n{df['quality'].value_counts()}")

    return df

# 3. 主程序
print("=" * 60)
print("WINE QUALITY CLASSIFICATION PROJECT")
print("=" * 60)

# Load data
wine_data = create_simulated_wine_data(3000)

print("\nDataset preview:")
print(wine_data.head())

# Data preprocessing
label_encoder = LabelEncoder()
wine_data['quality_encoded'] = label_encoder.fit_transform(wine_data['quality'])

print("\nLabel encoding:")
for i, cls in enumerate(label_encoder.classes_):
    print(f"  {cls} -> {i}")

# Prepare features and labels
features = ['country', 'points', 'price', 'variety']
X = wine_data[features]
y = wine_data['quality_encoded']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Create preprocessing pipeline
numeric_features = ['points', 'price']
categorical_features = ['country', 'variety']

# 创建预处理管道（包含标准化和独热编码）
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),# 数值特征标准化
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# 4. Train models
print("\n" + "="*60)
print("MODEL TRAINING")
print("="*60)

def train_and_evaluate_model(model_name, model_pipeline, param_grid):
    """Train and evaluate a single model"""
    print(f"\nTraining {model_name}...")
    start_time = time.time()

    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Training time: {training_time:.2f}s")

    return {
        'model': best_model,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'training_time': training_time,
        'best_params': grid_search.best_params_
    }

# Train Logistic Regression 构建逻辑回归模型管道
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor), # 数据预处理
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# 逻辑回归超参数搜索空间
lr_param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__solver': ['lbfgs', 'saga']
}

lr_results = train_and_evaluate_model("Logistic Regression", lr_pipeline, lr_param_grid)

# Train Random Forest 构建随机森林模型管道
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),# 数据预处理
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])

rf_param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None]
}

rf_results = train_and_evaluate_model("Random Forest", rf_pipeline, rf_param_grid)

# 5. 创建分开的图表
print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

# 图表1: 准确率对比
print("\nCreating Chart 1: Accuracy Comparison...")
plt.figure(figsize=(10, 6))

models = ['Logistic Regression', 'Random Forest']
accuracies = [lr_results['accuracy'], rf_results['accuracy']]
colors = ['#1f77b4', '#ff7f0e']

bars = plt.bar(models, accuracies, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
plt.ylabel('Accuracy', fontsize=14)
plt.ylim(0, 1.0)
plt.grid(True, alpha=0.3)

# 添加数值标签
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{acc:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('chart1_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()

# 图表2: 逻辑回归混淆矩阵
print("\nCreating Chart 2: Logistic Regression Confusion Matrix...")
plt.figure(figsize=(10, 8))

y_pred_lr = lr_results['y_pred']
cm_lr = confusion_matrix(y_test, y_pred_lr)

sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            cbar_kws={'label': 'Count'})
plt.title('Logistic Regression Confusion Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)

plt.tight_layout()
plt.savefig('chart2_lr_confusion.png', dpi=300, bbox_inches='tight')
plt.show()

# 图表3: 随机森林混淆矩阵
print("\nCreating Chart 3: Random Forest Confusion Matrix...")
plt.figure(figsize=(10, 8))

y_pred_rf = rf_results['y_pred']
cm_rf = confusion_matrix(y_test, y_pred_rf)

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            cbar_kws={'label': 'Count'})
plt.title('Random Forest Confusion Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)

plt.tight_layout()
plt.savefig('chart3_rf_confusion.png', dpi=300, bbox_inches='tight')
plt.show()

# 图表4: 特征重要性
print("\nCreating Chart 4: Feature Importance...")
plt.figure(figsize=(12, 8))

try:
    rf_model = rf_results['model'].named_steps['classifier']
    preprocessor = rf_results['model'].named_steps['preprocessor']

    # 获取特征名称
    cat_encoder = preprocessor.named_transformers_['cat']

    if hasattr(cat_encoder, 'get_feature_names_out'):
        cat_features = cat_encoder.get_feature_names_out(categorical_features)
    else:
        cat_features = []
        for i, col in enumerate(categorical_features):
            for cat in cat_encoder.categories_[i]:
                cat_features.append(f"{col}_{cat}")

    all_features = numeric_features + list(cat_features)
    feature_importance = rf_model.feature_importances_

    # 创建DataFrame
    importance_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True).tail(10)

    # 绘制水平条形图
    bars = plt.barh(importance_df['Feature'], importance_df['Importance'],
                   color='teal', edgecolor='black', alpha=0.7)
    plt.xlabel('Importance Score', fontsize=14)
    plt.title('Random Forest Feature Importance (Top 10)', fontsize=16, fontweight='bold')

    # 添加数值标签
    for bar, importance in zip(bars, importance_df['Importance']):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f' {width:.3f}', va='center', fontsize=11)

except Exception as e:
    plt.text(0.5, 0.5, 'Feature importance not available',
             ha='center', va='center', fontsize=12)
    plt.title('Feature Importance', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('chart4_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 图表5: F1分数对比
print("\nCreating Chart 5: F1 Score Comparison...")
plt.figure(figsize=(12, 8))

categories = label_encoder.classes_
f1_lr = f1_score(y_test, y_pred_lr, average=None)
f1_rf = f1_score(y_test, y_pred_rf, average=None)

x = np.arange(len(categories))
width = 0.35

bars_lr = plt.bar(x - width/2, f1_lr, width, label='Logistic Regression',
                 color='#1f77b4', edgecolor='black', alpha=0.8)
bars_rf = plt.bar(x + width/2, f1_rf, width, label='Random Forest',
                 color='#ff7f0e', edgecolor='black', alpha=0.8)

plt.xlabel('Quality Class', fontsize=14)
plt.ylabel('F1 Score', fontsize=14)
plt.title('F1 Score Comparison by Class', fontsize=16, fontweight='bold')
plt.xticks(x, categories, fontsize=12)
plt.legend(fontsize=12)
plt.ylim(0, 1.0)
plt.grid(True, alpha=0.3)

# 添加数值标签
for bars in [bars_lr, bars_rf]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('chart5_f1_scores.png', dpi=300, bbox_inches='tight')
plt.show()

# 图表6: 训练时间对比
print("\nCreating Chart 6: Training Time Comparison...")
plt.figure(figsize=(10, 6))

training_times = [lr_results['training_time'], rf_results['training_time']]

bars = plt.bar(models, training_times, color=colors,
              edgecolor='black', linewidth=2, alpha=0.8)
plt.title('Model Training Time Comparison', fontsize=16, fontweight='bold')
plt.ylabel('Training Time (seconds)', fontsize=14)
plt.grid(True, alpha=0.3)

# 添加数值标签
for bar, time_val in zip(bars, training_times):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{time_val:.2f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('chart6_training_time.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. 详细分析
print("\n" + "="*60)
print("DETAILED ANALYSIS")
print("="*60)

# 计算推理时间
print("\nInference Time Test:")
test_samples = X_test[:100]

start_time = time.time()
_ = lr_results['model'].predict(test_samples)
lr_inference_time = (time.time() - start_time) * 1000

start_time = time.time()
_ = rf_results['model'].predict(test_samples)
rf_inference_time = (time.time() - start_time) * 1000

print(f"Logistic Regression inference time: {lr_inference_time:.2f}ms (100 samples)")
print(f"Random Forest inference time: {rf_inference_time:.2f}ms (100 samples)")

# 性能总结表格
print("\n" + "="*60)
print("PERFORMANCE SUMMARY")
print("="*60)

summary_data = [
    ["Accuracy", f"{lr_results['accuracy']:.4f}", f"{rf_results['accuracy']:.4f}"],
    ["Training Time", f"{lr_results['training_time']:.2f}s", f"{rf_results['training_time']:.2f}s"],
    ["Inference Time", f"{lr_inference_time:.2f}ms", f"{rf_inference_time:.2f}ms"],
    ["Best Parameters", f"C={lr_results['best_params']['classifier__C']}",
     f"n_estimators={rf_results['best_params']['classifier__n_estimators']}"]
]

print("\n{:<15} {:<25} {:<25}".format("Metric", "Logistic Regression", "Random Forest"))
print("-" * 70)
for row in summary_data:
    print("{:<15} {:<25} {:<25}".format(row[0], row[1], row[2]))

# 7. 模型保存
print("\n" + "="*60)
print("MODEL SAVING")
print("="*60)

# 选择最佳模型
best_model_name = "Logistic Regression" if lr_results['accuracy'] >= rf_results['accuracy'] else "Random Forest"
best_model = lr_results['model'] if best_model_name == "Logistic Regression" else rf_results['model']

joblib.dump(best_model, 'best_wine_classifier.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print(f"\nBest model: {best_model_name}")
print("Model saved as 'best_wine_classifier.pkl'")
print("Label encoder saved as 'label_encoder.pkl'")

# 8. 预测示例
print("\n" + "="*60)
print("PREDICTION EXAMPLES")
print("="*60)

example_wines = pd.DataFrame({
    'country': ['France', 'Italy', 'USA'],
    'points': [95, 88, 81],
    'price': [150.0, 60.0, 25.0],
    'variety': ['Cabernet', 'Chardonnay', 'Merlot']
})

print("\nInput data:")
print(example_wines)

predictions = best_model.predict(example_wines)
probabilities = best_model.predict_proba(example_wines)

print("\nPrediction results:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    pred_label = label_encoder.inverse_transform([pred])[0]
    print(f"\nWine {i+1}:")
    print(f"  Predicted quality: {pred_label}")
    print(f"  Class probabilities:")
    for j, cls in enumerate(label_encoder.classes_):
        print(f"    {cls}: {prob[j]:.3f}")

# 9. 生成单个汇总图表
print("\nCreating final summary chart...")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 子图1: 准确率
bars1 = ax1.bar(models, accuracies, color=colors, edgecolor='black', alpha=0.8)
ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy')
ax1.set_ylim(0, 1.0)
ax1.grid(True, alpha=0.3)
for bar, acc in zip(bars1, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{acc:.3f}', ha='center', va='bottom', fontsize=11)

# 子图2: 训练时间
bars2 = ax2.bar(models, training_times, color=colors, edgecolor='black', alpha=0.8)
ax2.set_title('Training Time', fontsize=14, fontweight='bold')
ax2.set_ylabel('Time (seconds)')
ax2.grid(True, alpha=0.3)
for bar, time_val in zip(bars2, training_times):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{time_val:.2f}s', ha='center', va='bottom', fontsize=11)

# 子图3: F1分数
x = np.arange(len(categories))
width = 0.35
bars_lr = ax3.bar(x - width/2, f1_lr, width, label='Logistic Regression', color='#1f77b4', alpha=0.8)
bars_rf = ax3.bar(x + width/2, f1_rf, width, label='Random Forest', color='#ff7f0e', alpha=0.8)
ax3.set_title('F1 Scores by Class', fontsize=14, fontweight='bold')
ax3.set_xlabel('Quality Class')
ax3.set_ylabel('F1 Score')
ax3.set_xticks(x)
ax3.set_xticklabels(categories)
ax3.legend()
ax3.set_ylim(0, 1.0)
ax3.grid(True, alpha=0.3)

# 子图4: 特征重要性示例
if 'importance_df' in locals():
    top_features = importance_df.tail(5)
    bars4 = ax4.barh(top_features['Feature'], top_features['Importance'],
                    color='teal', edgecolor='black', alpha=0.7)
    ax4.set_title('Top 5 Important Features', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Importance Score')
else:
    ax4.text(0.5, 0.5, 'Feature importance data', ha='center', va='center', fontsize=12)
    ax4.set_title('Feature Importance', fontsize=14, fontweight='bold')

plt.suptitle('Wine Quality Classification - Model Comparison Summary', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('summary_chart.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nGenerated charts:")
print("1. chart1_accuracy.png - Model accuracy comparison")
print("2. chart2_lr_confusion.png - Logistic Regression confusion matrix")
print("3. chart3_rf_confusion.png - Random Forest confusion matrix")
print("4. chart4_feature_importance.png - Feature importance")
print("5. chart5_f1_scores.png - F1 score comparison")
print("6. chart6_training_time.png - Training time comparison")
print("7. summary_chart.png - Summary chart (all in one)")
print("\nGenerated files:")
print("1. best_wine_classifier.pkl - Best trained model")
print("2. label_encoder.pkl - Label encoder")