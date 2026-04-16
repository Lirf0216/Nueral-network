import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random

# 可视化设置
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 显示中文
plt.rcParams["axes.unicode_minus"] = False

# 数据加载与预处理,用于将原始图像数据转换为神经网络可处理的张量格式并进行标准化。
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# 加载MNIST数据集
# 训练集
trainset = torchvision.datasets.MNIST(
    root="./data", # 数据集保存的目录
    train=True, # 加载训练集（True=训练集，False=测试集）
    download=True, # 如果本地没有，则从网上下载，PyTorch的torchvision.datasets模块内置了下载功能
    transform=transform # 对图像进行预处理转换

)
#测试集
testset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

#创建了数据加载器,第一项为数据集对象，第二项意思为每批张图片，第三项的意思是打乱顺序
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
#同理
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

# 获取类别名称
class_names = [str(i) for i in range(10)]


# 定义添加椒盐噪声的函数
def add_salt_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    """
    给图像添加椒盐噪声
    salt_prob: 盐噪声概率（白色像素）
    pepper_prob: 椒噪声概率（黑色像素）
    """
    noisy_image = image.clone() #复制图像，保护原图
    height, width = noisy_image.shape[1], noisy_image.shape[2] #获取图像尺寸

    #Python图像的构成
    # noisy_image.shape = (C, H, W)
    #                     ↑  ↑  ↑
    #                   通道 高 宽

    # 添加盐噪声（白色）， 生成一个与图像尺寸相同的随机数矩阵，每个值在 [0, 1) 范围内，数值< salt_prob，将随机数与概率阈值比较，生成布尔掩码：
    salt_mask = torch.rand(height, width) < salt_prob
    #使用布尔掩码索引，将所有被选中的像素设为 1.0（白色）：
    noisy_image[0, salt_mask] = 1.0 # 直接覆盖原值

    # 添加椒噪声（黑色）
    pepper_mask = torch.rand(height, width) < pepper_prob
    noisy_image[0, pepper_mask] = -1.0

    return noisy_image


# 训练函数
# model: 要训练的神经网络模型, trainloader: 训练数据加载器, testloader: 测试数据加载器, epochs: 训练轮数（默认5轮）, lr: 学习率（默认0.01）
def train_model(model, trainloader, testloader, epochs=5, lr=0.01):
    # 初始化
    criterion = nn.CrossEntropyLoss() #创建一个交叉熵损失函数的实例，用于计算模型预测与真实标签之间的差异。
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    #     ↑          ↑           ↑              ↑        ↑
    #   变量名    优化器类型    要优化的参数    学习率    动量
    train_losses = []# 记录每轮训练损失
    test_accuracies = []# 记录每轮测试准确率

    # 一轮完整的训练和测试
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # 训练阶段
        for inputs, labels in trainloader:
            optimizer.zero_grad()# 将优化器中所有参数的梯度缓存清零
            outputs = model(inputs)  # 前向传播：将输入数据传入神经网络模型
            loss = criterion(outputs, labels) # 计算损失值：比较模型预测和真实标签的差异
            loss.backward() # 反向传播：自动计算损失相对于模型参数的梯度
            optimizer.step() # 参数更新：根据计算出的梯度更新模型参数
            running_loss += loss.item() # 累加损失值，用于后续计算整个epoch的平均损失

        avg_train_loss = running_loss / len(trainloader)# 计算当前epoch的平均训练损失
        train_losses.append(avg_train_loss)# 将当前epoch的平均损失记录到列表中

        # 测试阶段
        model.eval()# 设置模型为评估模式
        correct = 0 # 初始化正确预测的样本数
        total = 0 # 初始化总样本数

        with torch.no_grad():# 禁用梯度计算，节省内存
            for inputs, labels in testloader: # 遍历测试集的每个批次
                outputs = model(inputs) # 前向传播，获取模型预测
                _, predicted = torch.max(outputs.data, 1) # 获取预测类别（最大值索引）
                total += labels.size(0) # 累加当前批次的样本数
                correct += (predicted == labels).sum().item() # 累加正确预测数

        test_accuracy = 100 * correct / total # 计算测试准确率（百分比）
        test_accuracies.append(test_accuracy) # 记录准确率到列表

        print(
            f"Epoch {epoch+1}/{epochs}: 训练损失: {avg_train_loss:.4f}, 测试准确率: {test_accuracy:.2f}%"
        )

    return train_losses, test_accuracies


# 测试模型在噪声图像上的表现
def test_with_noise(model, testloader, noise_levels=[0.01, 0.02, 0.05, 0.1]):
    model.eval()  # 设置模型为评估模式
    results = {}  # 创建字典存储结果

    with torch.no_grad():  # 禁用梯度计算
        for noise_level in noise_levels:  # 遍历每个噪声水平
            correct = 0  # 初始化正确预测数
            total = 0    # 初始化总样本数

            for inputs, labels in testloader:  # 遍历测试批次
                # 添加噪声
                noisy_inputs = torch.stack(  # 将列表中的张量堆叠为批次
                    [
                        add_salt_pepper_noise(img, noise_level, noise_level)  # 添加椒盐噪声
                        for img in inputs  # 遍历批次中的每张图像
                    ]
                )

                outputs = model(noisy_inputs)  # 模型预测噪声图像
                _, predicted = torch.max(outputs.data, 1)  # 获取预测类别
                total += labels.size(0)  # 累加样本总数
                correct += (predicted == labels).sum().item()  # 累加正确数

            accuracy = 100 * correct / total  # 计算准确率
            results[noise_level] = accuracy  # 存储到结果字典
            print(f"噪声水平 {noise_level*100:.1f}%: 准确率 {accuracy:.2f}%")  # 打印结果

    return results  # 返回所有噪声水平的结果


# 作业一：对比Sigmoid和ReLU激活函数
print("=" * 50)
print("作业一：对比Sigmoid和ReLU激活函数")
print("=" * 50)

# 创建一个使用 Sigmoid 激活函数的神经网络模型
#model_sigmoid = nn.Sequential(
    # 第一层：将图像展平
    # 输入形状: (batch_size, 1, 28, 28) - 一批图片，每张是1通道28x28像素
    # 输出形状: (batch_size, 784) - 把每张图片的784个像素拉直成一维数组
    #nn.Flatten(),

    # 第二层：全连接层（也叫线性层）
    # 输入: 784个像素值
    # 输出: 256个特征值
    # 这一层有 784×256 个权重参数，负责将原始像素组合成更高层次的特征
    #nn.Linear(28 * 28, 256),

    # 第三层：Sigmoid 激活函数
    # 输入: 任意大小的数值（上一层的输出）
    # 输出: 压缩到 0 到 1 之间的数值
    # 作用: 引入非线性，让网络能学习更复杂的关系
    # 公式: f(x) = 1 / (1 + e^(-x))
    # 特点: 将大正数变成接近1，大负数变成接近0
    #nn.Sigmoid(),

    # 第四层：输出层（全连接层）
    # 输入: 256个特征值
    # 输出: 10个分数（对应数字0-9的得分）
    # 得分最高的位置就是模型预测的数字
    #nn.Linear(256, 10)
#)

# Sigmoid激活函数的模型
model_sigmoid = nn.Sequential(
    nn.Flatten(), nn.Linear(28 * 28, 256), nn.Sigmoid(), nn.Linear(256, 10)
)

# ReLU激活函数的模型
model_relu = nn.Sequential(
    nn.Flatten(), nn.Linear(28 * 28, 256), nn.ReLU(), nn.Linear(256, 10)
)

print("\n训练Sigmoid模型...")
train_losses_sigmoid, test_accuracies_sigmoid = train_model(
    model_sigmoid, trainloader, testloader
)

print("\n训练ReLU模型...")
train_losses_relu, test_accuracies_relu = train_model(
    model_relu, trainloader, testloader
)

# 作业二：去除隐藏层（只有输入层和输出层）
print("\n" + "=" * 50)  # 打印分隔线
print("作业二：去除隐藏层（只有输入层和输出层）")  # 打印标题
print("=" * 50)  # 打印分隔线

# 创建顺序模型
model_no_hidden = nn.Sequential(
    nn.Flatten(),  # 展平层：将二维图像展平为一维向量
    nn.Linear(28 * 28, 10)  # 线性层：直接从784个输入到10个输出
)

print("\n训练无隐藏层模型...")  # 打印训练开始提示
# 调用训练函数
train_losses_no_hidden, test_accuracies_no_hidden = train_model(
    model_no_hidden, trainloader, testloader  # 传入模型和数据加载器
)

# 作业三：增加隐藏层神经元数量
print("\n" + "=" * 50)
print("作业三：增加隐藏层神经元数量")
print("=" * 50)

# 原模型（256个神经元）
model_original = nn.Sequential(
    nn.Flatten(), nn.Linear(28 * 28, 256), nn.ReLU(), nn.Linear(256, 10)
)

# 增加神经元数量到512
model_more_neurons = nn.Sequential(
    nn.Flatten(), nn.Linear(28 * 28, 512), nn.ReLU(), nn.Linear(512, 10)
)

# 减少神经元数量到128
model_less_neurons = nn.Sequential(
    nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10)
)

print("\n训练原始模型（256个神经元）...")
train_losses_original, test_accuracies_original = train_model(
    model_original, trainloader, testloader
)

print("\n训练增加神经元模型（512个神经元）...")
train_losses_more, test_accuracies_more = train_model(
    model_more_neurons, trainloader, testloader
)

print("\n训练减少神经元模型（128个神经元）...")
train_losses_less, test_accuracies_less = train_model(
    model_less_neurons, trainloader, testloader
)

# 作业四：测试噪声影响
print("\n" + "=" * 50)
print("作业四：测试噪声对模型预测的影响")
print("=" * 50)

# 选择一个测试集图片样本进行可视化
test_images, test_labels = next(iter(testloader))
sample_image = test_images[0:1]  # 取第一个样本
sample_label = test_labels[0]

# 使用ReLU模型进行预测
model_relu.eval()
with torch.no_grad():
    output = model_relu(sample_image)
    _, predicted = torch.max(output.data, 1)
    original_prediction = predicted.item()

print(f"\n原始图片预测结果: 真实标签={sample_label}, 预测标签={original_prediction}")

# 测试不同噪声水平
noise_levels = [0.01, 0.02, 0.05, 0.1, 0.2]
noise_predictions = []

for noise_level in noise_levels:
    noisy_image = add_salt_pepper_noise(sample_image[0], noise_level, noise_level)

    with torch.no_grad():
        output = model_relu(noisy_image.unsqueeze(0))
        _, predicted = torch.max(output.data, 1)
        noise_predictions.append(predicted.item())

    print(f"噪声水平 {noise_level*100:.1f}%: 预测标签={predicted.item()}")

# 对整个测试集测试噪声影响
print("\n对整个测试集测试噪声影响（使用ReLU模型）:")
noise_results = test_with_noise(model_relu, testloader, noise_levels)

# 可视化结果
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. 激活函数对比
axes[0, 0].plot(range(1, 6), test_accuracies_sigmoid, "b-o", label="Sigmoid")
axes[0, 0].plot(range(1, 6), test_accuracies_relu, "r-o", label="ReLU")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("测试准确率 (%)")
axes[0, 0].set_title("激活函数对比 (Sigmoid vs ReLU)")
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. 隐藏层对比
axes[0, 1].bar(
    ["无隐藏层", "有隐藏层(ReLU)"],
    [test_accuracies_no_hidden[-1], test_accuracies_relu[-1]],
    color=["blue", "green"],
)
axes[0, 1].set_ylabel("测试准确率 (%)")
axes[0, 1].set_title("隐藏层对性能的影响")
axes[0, 1].grid(True, axis="y")

# 3. 神经元数量对比
axes[0, 2].bar(
    ["128神经元", "256神经元", "512神经元"],
    [test_accuracies_less[-1], test_accuracies_original[-1], test_accuracies_more[-1]],
    color=["lightblue", "blue", "darkblue"],
)
axes[0, 2].set_ylabel("测试准确率 (%)")
axes[0, 2].set_title("隐藏层神经元数量对比")
axes[0, 2].grid(True, axis="y")

# 4. 噪声对单个样本的影响
axes[1, 0].imshow(sample_image[0].squeeze(), cmap="gray")
axes[1, 0].set_title(f"原始图像\n真实: {sample_label}, 预测: {original_prediction}")
axes[1, 0].axis("off")

# 5. 不同噪声水平的图像
for i, noise_level in enumerate([0.01, 0.1]):
    noisy_image = add_salt_pepper_noise(sample_image[0], noise_level, noise_level)
    axes[1, i + 1].imshow(noisy_image.squeeze(), cmap="gray")
    axes[1, i + 1].set_title(
        f"噪声 {noise_level*100:.0f}%\n预测: {noise_predictions[i*2]}"
    )
    axes[1, i + 1].axis("off")

plt.tight_layout()
plt.show()

# 噪声对整体准确率的影响
fig2, ax2 = plt.subplots(figsize=(10, 6))
noise_levels_list = list(noise_results.keys())
accuracies_list = list(noise_results.values())

ax2.plot(noise_levels_list, accuracies_list, "r-o", linewidth=2, markersize=8)
ax2.set_xlabel("噪声水平")
ax2.set_ylabel("测试准确率 (%)")
ax2.set_title("噪声水平对模型准确率的影响")
ax2.grid(True)
ax2.set_xlim(0, max(noise_levels_list) + 0.01)

plt.tight_layout()
plt.show()

# 打印总结报告
print("\n" + "=" * 50)
print("实验总结报告")
print("=" * 50)

print(f"\n1. 激活函数对比:")
print(f"   Sigmoid - 最终准确率: {test_accuracies_sigmoid[-1]:.2f}%")
print(f"   ReLU    - 最终准确率: {test_accuracies_relu[-1]:.2f}%")
print(f"   结论: ReLU激活函数表现更好")

print(f"\n2. 隐藏层影响:")
print(f"   无隐藏层 - 最终准确率: {test_accuracies_no_hidden[-1]:.2f}%")
print(f"   有隐藏层 - 最终准确率: {test_accuracies_relu[-1]:.2f}%")
print(f"   结论: 隐藏层显著提升模型性能")

print(f"\n3. 神经元数量影响:")
print(f"   128神经元 - 最终准确率: {test_accuracies_less[-1]:.2f}%")
print(f"   256神经元 - 最终准确率: {test_accuracies_original[-1]:.2f}%")
print(f"   512神经元 - 最终准确率: {test_accuracies_more[-1]:.2f}%")
print(f"   结论: 适当增加神经元数量可以提升性能，但可能存在过拟合风险")

print(f"\n4. 噪声影响总结:")
for noise_level, accuracy in noise_results.items():
    print(f"   噪声水平 {noise_level*100:.1f}%: 准确率 {accuracy:.2f}%")
print(f"   结论: 噪声显著降低模型性能，噪声越大性能下降越明显")
