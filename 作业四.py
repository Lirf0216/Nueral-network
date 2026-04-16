import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 函数式接口
import torch.optim as optim  # 优化器
from torch.utils.data import DataLoader  # 数据加载器
from torchvision import datasets, transforms  # 视觉数据集和变换
import matplotlib.pyplot as plt  # 绘图库
import numpy as np  # 数值计算库
from tqdm import tqdm  # 进度条显示
import warnings  # 警告处理

warnings.filterwarnings("ignore")  # 忽略警告信息

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子
torch.manual_seed(42)  # 设置PyTorch随机种子
np.random.seed(42)  # 设置NumPy随机种子

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备
print(f"使用设备: {device}")  # 打印设备信息


# ============ 第一部分: 基础LeNet模型定义 ============
class LeNet(nn.Module):  # 定义LeNet类
    def __init__(self, activation="relu", pooling="max"):  # 初始化函数
        super(LeNet, self).__init__()  # 调用父类初始化

        # 根据参数选择激活函数和池化层
        if activation == "relu":  # 如果是ReLU
            self.activation = nn.ReLU()  # 使用ReLU激活函数
        elif activation == "sigmoid":  # 如果是Sigmoid
            self.activation = nn.Sigmoid()  # 使用Sigmoid激活函数
        else:  # 其他情况
            self.activation = nn.ReLU()  # 默认使用ReLU

        if pooling == "max":  # 如果是最大池化
            self.pool = nn.MaxPool2d(2, 2)  # 使用最大池化
        elif pooling == "avg":  # 如果是平均池化
            self.pool = nn.AvgPool2d(2, 2)  # 使用平均池化
        else:  # 其他情况
            self.pool = nn.MaxPool2d(2, 2)  # 默认使用最大池化

        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)  # 第一层卷积
        self.conv2 = nn.Conv2d(6, 16, 5)  # 第二层卷积

        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 第一个全连接层
        self.fc2 = nn.Linear(120, 84)  # 第二个全连接层
        self.fc3 = nn.Linear(84, 10)  # 输出层

        # 用于存储中间特征图的钩子
        self.feature_maps = {}  # 特征图字典
        self.hooks = []  # 钩子列表

    def forward(self, x):  # 前向传播函数
        # 第一层卷积
        x = self.conv1(x)  # 卷积操作
        x = self.activation(x)  # 激活函数
        x = self.pool(x)  # 池化操作

        # 存储第一层特征图
        if not hasattr(self, "feature_map1"):  # 如果没有特征图1属性
            self.feature_map1 = x.clone().detach()  # 克隆并存储特征图

        # 第二层卷积
        x = self.conv2(x)  # 卷积操作
        x = self.activation(x)  # 激活函数
        x = self.pool(x)  # 池化操作

        # 存储第二层特征图
        if not hasattr(self, "feature_map2"):  # 如果没有特征图2属性
            self.feature_map2 = x.clone().detach()  # 克隆并存储特征图

        # 展平
        x = x.view(-1, 16 * 5 * 5)  # 展平为向量

        # 全连接层
        x = self.activation(self.fc1(x))  # 第一个全连接层
        x = self.activation(self.fc2(x))  # 第二个全连接层
        x = self.fc3(x)  # 输出层

        return x  # 返回输出

    def register_hooks(self):  # 注册钩子函数
        """注册钩子来捕获各层的输出"""

        def get_hook(name):  # 定义钩子获取函数
            def hook(module, input, output):  # 钩子函数
                self.feature_maps[name] = output.detach()  # 存储输出特征图

            return hook  # 返回钩子函数

        # 注册钩子
        self.hooks.append(
            self.conv1.register_forward_hook(get_hook("conv1"))
        )  # 注册conv1钩子
        self.hooks.append(
            self.conv2.register_forward_hook(get_hook("conv2"))
        )  # 注册conv2钩子

    def remove_hooks(self):  # 移除钩子函数
        """移除钩子"""
        for hook in self.hooks:  # 遍历所有钩子
            hook.remove()  # 移除钩子
        self.hooks = []  # 清空钩子列表


# ============ 第二部分: MLP模型定义（用于作业三） ============
class MLP(nn.Module):  # 定义MLP类
    def __init__(self):  # 初始化函数
        super(MLP, self).__init__()  # 调用父类初始化
        self.fc1 = nn.Linear(28 * 28, 64)  # 第一个全连接层
        self.fc2 = nn.Linear(64, 32)  # 第二个全连接层
        self.fc3 = nn.Linear(32, 10)  # 输出层
        self.dropout = nn.Dropout(0.5)  # Dropout层

    def forward(self, x):  # 前向传播函数
        x = x.view(-1, 28 * 28)  # 展平输入
        x = F.relu(self.fc1(x))  # 第一个全连接层+ReLU
        x = self.dropout(x)  # Dropout
        x = F.relu(self.fc2(x))  # 第二个全连接层+ReLU
        x = self.dropout(x)  # Dropout
        x = self.fc3(x)  # 输出层
        return x  # 返回输出


# ============ 第三部分: 训练和测试函数 ============
def train(model, device, train_loader, optimizer, criterion, epoch):  # 训练函数
    model.train()  # 设置为训练模式
    train_loss = 0  # 训练损失
    correct = 0  # 正确预测数
    total = 0  # 总样本数

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")  # 创建进度条
    for batch_idx, (data, target) in enumerate(pbar):  # 遍历数据
        data, target = data.to(device), target.to(device)  # 数据移到设备

        optimizer.zero_grad()  # 梯度清零
        output = model(data)  # 前向传播
        loss = criterion(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        train_loss += loss.item()  # 累加损失
        _, predicted = output.max(1)  # 获取预测类别
        total += target.size(0)  # 累加样本数
        correct += predicted.eq(target).sum().item()  # 累加正确数

        pbar.set_postfix(
            {"Loss": f"{loss.item():.4f}", "Acc": f"{100.*correct/total:.2f}%"}
        )  # 更新进度条

    return (
        train_loss / len(train_loader),
        100.0 * correct / total,
    )  # 返回平均损失和准确率


def test(model, device, test_loader, criterion):  # 测试函数
    model.eval()  # 设置为评估模式
    test_loss = 0  # 测试损失
    correct = 0  # 正确预测数
    total = 0  # 总样本数

    with torch.no_grad():  # 不计算梯度
        for data, target in test_loader:  # 遍历测试数据
            data, target = data.to(device), target.to(device)  # 数据移到设备
            output = model(data)  # 前向传播
            test_loss += criterion(output, target).item()  # 计算损失
            _, predicted = output.max(1)  # 获取预测类别
            total += target.size(0)  # 累加样本数
            correct += predicted.eq(target).sum().item()  # 累加正确数

    test_loss /= len(test_loader)  # 计算平均损失
    accuracy = 100.0 * correct / total  # 计算准确率

    return test_loss, accuracy  # 返回损失和准确率


# ============ 第四部分: 可视化特征图（作业一）修复版本 ============
def visualize_feature_maps(model, images, labels, target_label=3):  # 特征图可视化函数
    """
    可视化同一标签的两张图片在每一层卷积后的特征图
    """
    model.eval()  # 设置为评估模式
    model.register_hooks()  # 注册钩子

    # 找到目标标签的两张图片
    target_indices = []  # 目标索引列表
    for i in range(len(images)):  # 遍历图片
        if labels[i] == target_label:  # 如果标签匹配
            target_indices.append(i)  # 添加到列表
        if len(target_indices) >= 2:  # 如果找到足够多的图片
            break  # 停止搜索

    if len(target_indices) < 2:  # 如果没有足够多的图片
        print(f"没有找到足够多的标签为{target_label}的图片")  # 打印提示
        return  # 返回

    # 准备画图 - 使用更大的布局
    fig, axes = plt.subplots(4, 6, figsize=(15, 10))  # 创建4行6列的子图
    fig.suptitle(f"特征图可视化 - 数字 {target_label}", fontsize=16)  # 设置标题

    # 清除所有子图
    for i in range(4):
        for j in range(6):
            axes[i, j].axis("off")  # 关闭所有坐标轴

    # 显示原始图片
    for i, idx in enumerate(target_indices):  # 遍历目标图片
        img = images[idx].squeeze().numpy()  # 获取图片数据
        row_start = i * 2  # 计算起始行
        axes[row_start, 0].imshow(img, cmap="gray")  # 显示原始图片
        axes[row_start, 0].set_title(f"原始图片 {i+1}", fontsize=10)  # 设置标题
        axes[row_start, 0].axis("on")  # 打开坐标轴
        axes[row_start, 0].set_xticks([])  # 隐藏x轴刻度
        axes[row_start, 0].set_yticks([])  # 隐藏y轴刻度

    # 前向传播获取特征图
    with torch.no_grad():  # 不计算梯度
        for i, idx in enumerate(target_indices):  # 遍历目标图片
            img_tensor = images[idx].unsqueeze(0).to(device)  # 准备输入张量
            _ = model(img_tensor)  # 前向传播

            row_start = i * 2  # 计算起始行

            # 获取conv1的特征图
            if "conv1" in model.feature_maps:  # 如果有conv1特征图
                conv1_features = (
                    model.feature_maps["conv1"][0].cpu().numpy()
                )  # 获取特征图
                # 显示前6个通道
                for j in range(min(6, conv1_features.shape[0])):  # 遍历前6个通道
                    axes[row_start, j].imshow(
                        conv1_features[j], cmap="viridis"
                    )  # 显示特征图
                    axes[row_start, j].axis("on")  # 打开坐标轴
                    axes[row_start, j].set_xticks([])  # 隐藏x轴刻度
                    axes[row_start, j].set_yticks([])  # 隐藏y轴刻度
                    if j == 0:  # 如果是第一个通道
                        axes[row_start, j].set_title(
                            f"图片{i+1} Conv1", fontsize=10
                        )  # 设置标题
                    else:  # 其他通道
                        axes[row_start, j].set_title(f"通道{j}", fontsize=8)  # 设置标题

            # 获取conv2的特征图
            if "conv2" in model.feature_maps:  # 如果有conv2特征图
                conv2_features = (
                    model.feature_maps["conv2"][0].cpu().numpy()
                )  # 获取特征图
                # 显示前6个通道
                for j in range(min(6, conv2_features.shape[0])):  # 遍历前6个通道
                    axes[row_start + 1, j].imshow(
                        conv2_features[j], cmap="viridis"
                    )  # 显示特征图
                    axes[row_start + 1, j].axis("on")  # 打开坐标轴
                    axes[row_start + 1, j].set_xticks([])  # 隐藏x轴刻度
                    axes[row_start + 1, j].set_yticks([])  # 隐藏y轴刻度
                    if j == 0:  # 如果是第一个通道
                        axes[row_start + 1, j].set_title(
                            f"图片{i+1} Conv2", fontsize=10
                        )  # 设置标题
                    else:  # 其他通道
                        axes[row_start + 1, j].set_title(
                            f"通道{j}", fontsize=8
                        )  # 设置标题

    plt.tight_layout()  # 调整布局
    plt.subplots_adjust(top=0.92)  # 调整顶部空间
    plt.show()  # 显示图形
    model.remove_hooks()  # 移除钩子


# ============ 第五部分: 主程序 ============
def main():  # 主函数
    # 超参数
    batch_size = 64  # 批量大小
    learning_rate = 0.001  # 学习率
    epochs = 10  # 训练轮数

    # 数据加载和预处理
    transform = transforms.Compose(
        [  # 数据预处理组合
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize((0.1307,), (0.3081,)),  # 标准化
        ]
    )

    # 加载MNIST数据集
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )  # 训练集
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )  # 测试集

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )  # 训练数据加载器
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )  # 测试数据加载器

    # ============ 作业一: 可视化特征图 ============
    print("\n" + "=" * 60)  # 分隔线
    print("作业一: 可视化特征图")  # 标题
    print("=" * 60)  # 分隔线

    # 创建并训练一个LeNet模型
    lenet_model = LeNet(activation="relu", pooling="max").to(device)  # 创建LeNet模型
    criterion = nn.CrossEntropyLoss()  # 损失函数
    optimizer = optim.Adam(lenet_model.parameters(), lr=learning_rate)  # 优化器

    # 快速训练（为了演示，我们只训练几个epoch）
    print("训练LeNet用于特征图可视化...")  # 打印提示
    for epoch in range(3):  # 训练3轮
        train(lenet_model, device, train_loader, optimizer, criterion, epoch)  # 训练

    # 获取测试集中的图片
    test_data_iter = iter(test_loader)  # 创建测试数据迭代器
    test_images, test_labels = next(test_data_iter)  # 获取测试数据

    # 可视化特征图（选择数字3）
    visualize_feature_maps(
        lenet_model, test_images, test_labels, target_label=3
    )  # 可视化特征图

    # ============ 作业二: 比较不同激活函数和池化层 ============
    print("\n" + "=" * 60)  # 分隔线
    print("作业二: 比较不同激活函数和池化层")  # 标题
    print("=" * 60)  # 分隔线

    configs = [  # 配置列表
        ("ReLU + MaxPool", "relu", "max"),  # 配置1
        ("Sigmoid + AvgPool", "sigmoid", "avg"),  # 配置2
        ("ReLU + AvgPool", "relu", "avg"),  # 配置3
        ("Sigmoid + MaxPool", "sigmoid", "max"),  # 配置4
    ]

    results = {}  # 结果字典

    for config_name, activation, pooling in configs:  # 遍历配置
        print(f"\n训练 {config_name} 配置...")  # 打印提示
        model = LeNet(activation=activation, pooling=pooling).to(device)  # 创建模型
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 优化器

        best_acc = 0  # 最佳准确率
        for epoch in range(epochs):  # 训练多个轮次
            train_loss, train_acc = train(
                model, device, train_loader, optimizer, criterion, epoch
            )  # 训练
            test_loss, test_acc = test(model, device, test_loader, criterion)  # 测试

            if test_acc > best_acc:  # 如果准确率更高
                best_acc = test_acc  # 更新最佳准确率

        results[config_name] = best_acc  # 存储结果
        print(f"{config_name} 最佳测试准确率: {best_acc:.2f}%")  # 打印结果

    # 打印比较结果
    print("\n不同配置的准确率比较:")  # 标题
    for config_name, acc in results.items():  # 遍历结果
        print(f"{config_name}: {acc:.2f}%")  # 打印每个配置的准确率

    # ============ 作业三: 与MLP比较 ============
    print("\n" + "=" * 60)  # 分隔线
    print("作业三: 与MLP比较")  # 标题
    print("=" * 60)  # 分隔线

    # 计算LeNet的参数数量
    lenet = LeNet().to(device)  # 创建LeNet
    lenet_params = sum(p.numel() for p in lenet.parameters())  # 计算参数数量
    print(f"LeNet参数数量: {lenet_params}")  # 打印参数数量

    # 训练MLP
    print("\n训练MLP...")  # 打印提示
    mlp_model = MLP().to(device)  # 创建MLP模型
    mlp_params = sum(p.numel() for p in mlp_model.parameters())  # 计算参数数量
    print(f"MLP参数数量: {mlp_params}")  # 打印参数数量

    optimizer_mlp = optim.Adam(mlp_model.parameters(), lr=learning_rate)  # MLP优化器

    mlp_best_acc = 0  # MLP最佳准确率
    for epoch in range(epochs):  # 训练多个轮次
        train_loss, train_acc = train(
            mlp_model, device, train_loader, optimizer_mlp, criterion, epoch
        )  # 训练
        test_loss, test_acc = test(mlp_model, device, test_loader, criterion)  # 测试

        if test_acc > mlp_best_acc:  # 如果准确率更高
            mlp_best_acc = test_acc  # 更新最佳准确率

    # 训练LeNet进行比较
    print("\n训练LeNet进行比较...")  # 打印提示
    lenet_model2 = LeNet().to(device)  # 创建LeNet模型
    optimizer_lenet = optim.Adam(
        lenet_model2.parameters(), lr=learning_rate
    )  # LeNet优化器

    lenet_best_acc = 0  # LeNet最佳准确率
    for epoch in range(epochs):  # 训练多个轮次
        train_loss, train_acc = train(
            lenet_model2, device, train_loader, optimizer_lenet, criterion, epoch
        )  # 训练
        test_loss, test_acc = test(lenet_model2, device, test_loader, criterion)  # 测试

        if test_acc > lenet_best_acc:  # 如果准确率更高
            lenet_best_acc = test_acc  # 更新最佳准确率

    print(f"\n模型比较结果:")  # 标题
    print(f"MLP (参数: {mlp_params}) 最佳测试准确率: {mlp_best_acc:.2f}%")  # MLP结果
    print(
        f"LeNet (参数: {lenet_params}) 最佳测试准确率: {lenet_best_acc:.2f}%"
    )  # LeNet结果

    # ============ 作业四: FashionMNIST图像识别 ============
    print("\n" + "=" * 60)  # 分隔线
    print("作业四: FashionMNIST图像识别")  # 标题
    print("=" * 60)  # 分隔线

    # 加载FashionMNIST数据集
    fashion_transform = transforms.Compose(
        [  # 数据预处理组合
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize((0.2860,), (0.3530,)),  # 标准化
        ]
    )

    fashion_train_dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=fashion_transform
    )  # 训练集
    fashion_test_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=fashion_transform
    )  # 测试集

    fashion_train_loader = DataLoader(
        fashion_train_dataset, batch_size=batch_size, shuffle=True
    )  # 训练数据加载器
    fashion_test_loader = DataLoader(
        fashion_test_dataset, batch_size=batch_size, shuffle=False
    )  # 测试数据加载器

    # 创建改进的LeNet模型（增加通道数，添加Dropout）
    class ImprovedLeNet(nn.Module):  # 改进的LeNet类
        def __init__(self):  # 初始化函数
            super(ImprovedLeNet, self).__init__()  # 调用父类初始化
            self.conv1 = nn.Conv2d(1, 32, 5, padding=2)  # 第一层卷积，更多通道
            self.conv2 = nn.Conv2d(32, 64, 5)  # 第二层卷积，更多通道
            self.fc1 = nn.Linear(64 * 5 * 5, 256)  # 第一个全连接层，更多神经元
            self.fc2 = nn.Linear(256, 128)  # 第二个全连接层
            self.fc3 = nn.Linear(128, 10)  # 输出层
            self.dropout = nn.Dropout(0.5)  # Dropout层
            self.pool = nn.MaxPool2d(2, 2)  # 最大池化层

        def forward(self, x):  # 前向传播函数
            x = F.relu(self.conv1(x))  # 第一层卷积+ReLU
            x = self.pool(x)  # 池化
            x = F.relu(self.conv2(x))  # 第二层卷积+ReLU
            x = self.pool(x)  # 池化
            x = x.view(-1, 64 * 5 * 5)  # 展平
            x = F.relu(self.fc1(x))  # 第一个全连接层+ReLU
            x = self.dropout(x)  # Dropout
            x = F.relu(self.fc2(x))  # 第二个全连接层+ReLU
            x = self.dropout(x)  # Dropout
            x = self.fc3(x)  # 输出层
            return x  # 返回输出

    # 训练改进的LeNet
    print("训练改进的LeNet在FashionMNIST上...")  # 打印提示
    improved_lenet = ImprovedLeNet().to(device)  # 创建改进的LeNet
    optimizer_improved = optim.Adam(
        improved_lenet.parameters(), lr=0.0005
    )  # 优化器，更低学习率
    scheduler = optim.lr_scheduler.StepLR(
        optimizer_improved, step_size=5, gamma=0.5
    )  # 学习率调度器

    best_fashion_acc = 0  # FashionMNIST最佳准确率
    for epoch in range(15):  # 更多训练轮次
        train_loss, train_acc = train(
            improved_lenet,
            device,
            fashion_train_loader,
            optimizer_improved,
            criterion,
            epoch,
        )  # 训练
        test_loss, test_acc = test(
            improved_lenet, device, fashion_test_loader, criterion
        )  # 测试
        scheduler.step()  # 更新学习率

        if test_acc > best_fashion_acc:  # 如果准确率更高
            best_fashion_acc = test_acc  # 更新最佳准确率
            torch.save(
                improved_lenet.state_dict(), "best_fashion_lenet.pth"
            )  # 保存模型权重

        print(f"Epoch {epoch+1}: 测试准确率: {test_acc:.2f}%")  # 打印每轮结果

        if test_acc >= 80:  # 如果达到目标准确率
            print(f"达到目标准确率80%! 当前准确率: {test_acc:.2f}%")  # 打印成功信息
            break  # 停止训练

    print(f"\nFashionMNIST最佳测试准确率: {best_fashion_acc:.2f}%")  # 打印最终结果

    if best_fashion_acc >= 80:  # 如果达到目标
        print("成功达到80%以上的准确率!")  # 打印成功信息
    else:  # 如果未达到目标
        print("未达到80%的准确率，尝试以下方法改进:")  # 打印改进建议
        print("1. 增加训练epochs")  # 建议1
        print("2. 使用数据增强")  # 建议2
        print("3. 调整学习率")  # 建议3
        print("4. 使用更深的网络结构")  # 建议4

    # ============ 总结 ============
    print("\n" + "=" * 60)  # 分隔线
    print("作业总结")  # 标题
    print("=" * 60)  # 分隔线
    print("1. 特征图可视化: 已完成")  # 总结1
    print("2. 激活函数和池化层比较: 已完成")  # 总结2
    print("3. 与MLP比较: 已完成")  # 总结3
    print(
        "4. FashionMNIST识别: 已完成，最佳准确率: {:.2f}%".format(best_fashion_acc)
    )  # 总结4

    # 绘制配置比较图
    plt.figure(figsize=(10, 6))  # 创建图形
    config_names = list(results.keys())  # 获取配置名称
    acc_values = list(results.values())  # 获取准确率值

    bars = plt.bar(
        config_names, acc_values, color=["blue", "orange", "green", "red"]
    )  # 绘制柱状图
    plt.axhline(
        y=mlp_best_acc,
        color="purple",
        linestyle="--",
        label=f"MLP ({mlp_best_acc:.1f}%)",
    )  # MLP基准线
    plt.axhline(
        y=lenet_best_acc,
        color="cyan",
        linestyle="--",
        label=f"LeNet ({lenet_best_acc:.1f}%)",
    )  # LeNet基准线
    plt.axhline(
        y=best_fashion_acc,
        color="brown",
        linestyle="--",
        label=f"FashionLeNet ({best_fashion_acc:.1f}%)",
    )  # FashionMNIST基准线

    plt.xlabel("模型配置")  # X轴标签
    plt.ylabel("准确率 (%)")  # Y轴标签
    plt.title("不同模型配置的准确率比较")  # 图形标题
    plt.ylim(0, 100)  # Y轴范围
    plt.legend()  # 显示图例
    plt.grid(True, alpha=0.3)  # 显示网格
    plt.tight_layout()  # 调整布局
    plt.show()  # 显示图形


if __name__ == "__main__":  # 主程序入口
    main()  # 调用主函数
