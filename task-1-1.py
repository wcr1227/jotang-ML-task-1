
# 申明：代码由ai辅助书写，大部分疑问与解释由本人书写

# ============ 1. 导入库 ============
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


#设置中文字体（补充修改）

# ============ 1.1 让图里能显示中文 ============

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",   # Windows 微软雅黑
    "SimHei",            # Windows 黑体
    "PingFang SC",       # Mac 苹方
    "Noto Sans CJK SC",  # Linux 常用
    "WenQuanYi Zen Hei", # Linux 文泉驿
]
plt.rcParams["axes.unicode_minus"] = False   # 让负号正常显示，不然负号也会变方框


# ============ 2. 固定随机种子============
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
# 选择设备
set_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("使用设备：", device)

# ============ 3. 生成数据 ============
X, y = make_moons(n_samples=2000, noise=0.2, random_state=42)
# n_samples=2000 ：总共生成2000个样本
# noise=0.2 ：给数据加高斯噪声，数值越大点越散乱，0就是完美光滑月牙
# random_state=42 ：固定随机种子，每次运行代码生成一模一样的数据



print("数据形状：", X.shape, y.shape)   # (2000, 2) (2000,)

# 可视化原始数据
plt.figure(figsize=(6, 5))
plt.scatter(X[y==0, 0], X[y==0, 1], c='tab:blue',  label='类别 0', alpha=0.6)
# 这里说明一下   X[行筛选，列索引]
# 行： y==0  → 只保留标签=0的那些样本（行）
# 列： 0  → 取选中样本的第0列（第0号特征）


plt.scatter(X[y==1, 0], X[y==1, 1], c='tab:orange', label='类别 1', alpha=0.6)
plt.title("make_moons 原始数据")
plt.legend() # 显示图例（右上角，说明蓝、橙分别代表哪一类）
plt.grid(alpha=0.3) # 画淡淡的网格线，方便看坐标
plt.show()

# ============ 4. 划分 训练/验证/测试 ============
# 先分 70% 训练，30% 临时
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
# 临时集再对半分 → 15% 验证，15% 测试
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y, random_state=42)

print(f"训练集 {len(X_train)}，验证集 {len(X_val)}，测试集 {len(X_test)}")

# 标准化

# 这里是从学习资料LR_jupyter.ipynb的代码中学习摘录的

# 为什么要标准化？
# 不同特征的量纲和取值范围可能不同：
# 例如：
# GRE Score 大约是 260 ~ 340
# TOEFL Score 大约是 90 ~ 120
# CGPA 大约是 6 ~ 10
# Research 只有 0 或 1
#
# 如果不标准化，不同特征的数值尺度差别很大，
# 这可能导致模型训练不稳定，或者收敛速度变慢。
#
# 标准化的作用：
# 让每个特征变成均值约为 0、标准差约为 1 的分布。
#
# 标准化后，每个特征都站在一个比较公平的尺度上，
# 梯度下降时更新参数会更加稳定。

# 创建一个 StandardScaler 实例
# StandardScaler 是 sklearn.preprocessing 模块中的一个工具
# 用于将特征缩放为均值为 0，标准差为 1 的分布
# 也称为 Z-score 标准化



scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# 转成 PyTorch 的 tensor
# PyTorch 的神经网络不能直接接收 numpy，必须转换成 PyTorch 专属的数据类型：Tensor（张量）
def to_tensor(X, y):
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
# dtype=torch.float32 ：指定数据类型为32位浮点数
# X是特征（两个坐标值），神经网络做矩阵乘法，必须是浮点数
# PyTorch的CrossEntropyLoss交叉熵损失函数，要求标签必须是long类型

Xtr, ytr = to_tensor(X_train, y_train) # 训练集特征张量，训练集标签张量
Xva, yva = to_tensor(X_val,   y_val) # 验证集
Xte, yte = to_tensor(X_test,  y_test) # 测试集

# TensorDataset(Xtr, ytr) ：把特征张量与标签张量打包成数据集对象
train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=32, shuffle=True)
# batch_size=32:每次取32个样本作为一个批次（batch）
# shuffle=True ：每一轮epoch（完整遍历全部训练集一遍）开始前，打乱全部样本顺序
val_loader   = DataLoader(TensorDataset(Xva, yva), batch_size=64)
test_loader  = DataLoader(TensorDataset(Xte, yte), batch_size=64)

# 疑问：为什么训练集 batch_size=32，验证/测试集 batch_size=64?
# 训练要小batch帮助模型更好学习,验证测试不更新参数，显存富余，可以用更大batch提速


# ============ 5. 定义 MLP ============
class MLP(nn.Module): # 新建一个 MLP 类，继承PyTorch的 nn.Module ，所有神经网络都必须继承这个父类
    def __init__(self, in_dim=2, hidden=64, depth=2, n_class=2):
        # in_dim=2 ：输入维度=2,匹配make_moons，每个样本2个特征
        # hidden=64 ：隐藏层神经元个数，每层64个神经元
        # depth=2 ：隐藏层的数量，设置2层隐藏层
        # n_class=2 ：分类类别数，二分类，输出2个值

        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):                 
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            # nn.Linear(d, hidden) ：全连接层（线性层）
            # nn.ReLU() ：激活函数，给网络加入非线性能力
            d = hidden # 更新维度
        layers += [nn.Linear(d, n_class)] # 输出层    
        self.net = nn.Sequential(*layers) # nn.Sequential ：把列表里所有层按顺序串起来，打包成一个完整网络
    
    # 前向传播
    def forward(self, x):
        return self.net(x)

model = MLP(hidden=64, depth=2).to(device)
print(model)

# ============ 6. 训练配置 ============
# 定义损失函数和优化器
# nn.CrossEntropyLoss() ：交叉熵损失，用于分类任务
criterion = nn.CrossEntropyLoss()
# Adam优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # 学习率
EPOCHS = 200 # 一共循环完整遍历训练集200次

# 记录曲线用
history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}

# 定义评估函数
# 用来在训练过程中，评估模型在验证集（或者训练结束后评估测试集）上的损失和准确率，看看模型效果好不好
def evaluate(model, loader):
    model.eval()
    total_loss, correct, n = 0, 0, 0 
    with torch.no_grad():
        #关闭梯度记录功能
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            total_loss += criterion(out, yb).item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()
            n += len(xb)
    return total_loss / n, correct / n

# ============ 7. 训练循环 ============
for epoch in range(EPOCHS):
    model.train()
    tl, tc, tn = 0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        out = model(xb)
        # 交叉熵损失函数
        loss = criterion(out, yb)
        
        # 反向传播  很重要！！！
        loss.backward()
        
        # 参数更新
        optimizer.step()
        tl += loss.item() * len(xb)
        tc += (out.argmax(1) == yb).sum().item()
        tn += len(xb)
    
    # 训练循环里的评估
    train_loss, train_acc = tl/tn, tc/tn
    val_loss, val_acc = evaluate(model, val_loader)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1:3d} | train_loss {train_loss:.4f} acc {train_acc:.3f} "
              f"| val_loss {val_loss:.4f} acc {val_acc:.3f}")

# ============ 8. 测试 & 保存/加载 ============
# 最终的评估
test_loss, test_acc = evaluate(model, test_loader)
print(f"\n测试集:loss {test_loss:.4f}, acc {test_acc:.4f}")

# 保存模型权重
torch.save(model.state_dict(), 'mlp.pth')

# 加载：要先重建同样结构的模型
model2 = MLP(hidden=64, depth=2).to(device)
model2.load_state_dict(torch.load('mlp.pth'))
model2.eval()
print("模型加载成功 ✔")

# ============ 9. 画 loss / acc 曲线 ============
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='train')
plt.plot(history['val_loss'],   label='val')
plt.xlabel('epoch'); plt.ylabel('loss'); plt.title('Loss 曲线'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='train')
plt.plot(history['val_acc'],   label='val')
plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.title('Accuracy 曲线'); plt.legend()
plt.tight_layout(); plt.show()

# ============ 10. 画决策边界 ============
# 疑问:这是在干嘛？
# 可以简单理解为把平面上密密麻麻的所有点全部喂给模型预测，然后根据预测结果上色，就能看出模型划分两类的区域
# 目的：判断模型是否欠拟合或者过拟合
def plot_decision_boundary(model, X, y, scaler, title='决策边界'):
    model.eval()
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid = scaler.transform(grid)                     # 记得标准化！
    grid = torch.tensor(grid, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(grid).argmax(1).cpu().numpy().reshape(xx.shape)

    plt.figure(figsize=(6,5))
    plt.contourf(xx, yy, pred, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[y==0,0], X[y==0,1], c='tab:blue',   edgecolor='k', s=20)
    plt.scatter(X[y==1,0], X[y==1,1], c='tab:orange', edgecolor='k', s=20)
    plt.title(title); plt.show()

# 用原始（未标准化）数据画背景
plot_decision_boundary(model, X_train, y_train, scaler, '训练集决策边界')

# ============ 11. 混淆矩阵 ============
# 疑问：这是在干嘛？
# 统计各类别预测对错的数量（量化错误）(ai说的，感觉不太懂)
model.eval()
with torch.no_grad():
    preds = model(Xte.to(device)).argmax(1).cpu().numpy()
cm = confusion_matrix(y_test, preds)
ConfusionMatrixDisplay(cm, display_labels=['0','1']).plot(cmap='Blues')
plt.title('测试集混淆矩阵'); plt.show()

# ============ 12. 找出分错的样本 ============
wrong_idx = np.where(preds != y_test)[0]
print(f"分错了 {len(wrong_idx)} 个样本，前 5 个：")
for i in wrong_idx[:5]:
    print(f"  坐标 {X_test[i]}，真实 {y_test[i]}，预测 {preds[i]}")
