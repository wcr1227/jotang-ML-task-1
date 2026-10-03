# 下面这段代码是在 task-1-1.py 中定义的 MLP 模型和训练循环的基础上，增加了一个新的功能：
# 在训练过程中保存模型的最佳状态，并在训练结束后加载该最佳模型进行评估
# ============ 1. 导入库 ============
import copy
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

# ============ 2. 固定随机种子（保证学长能复现）============
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("使用设备：", device)

# ============ 3. 生成数据 ============
X, y = make_moons(n_samples=2000, noise=0.2, random_state=42)
print("数据形状：", X.shape, y.shape)   # (2000, 2) (2000,)

# 可视化原始数据
plt.figure(figsize=(6, 5))
plt.scatter(X[y==0, 0], X[y==0, 1], c='tab:blue',  label='类别 0', alpha=0.6)
plt.scatter(X[y==1, 0], X[y==1, 1], c='tab:orange', label='类别 1', alpha=0.6)
plt.title("make_moons 原始数据")
plt.legend(); plt.grid(alpha=0.3)
plt.show()

# ============ 4. 划分 训练/验证/测试 ============
# 先分 70% 训练，30% 临时
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
# 临时集再对半分 → 15% 验证，15% 测试
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y, random_state=42)

print(f"训练集 {len(X_train)}，验证集 {len(X_val)}，测试集 {len(X_test)}")

# 标准化：只在训练集上 fit，避免数据泄漏！
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# 转成 PyTorch 的 tensor
def to_tensor(X, y):
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

Xtr, ytr = to_tensor(X_train, y_train)
Xva, yva = to_tensor(X_val,   y_val)
Xte, yte = to_tensor(X_test,  y_test)

train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(Xva, yva), batch_size=64)
test_loader  = DataLoader(TensorDataset(Xte, yte), batch_size=64)

# ============ 5. 定义 MLP ============
class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=64, depth=2, n_class=2):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):                 # 至少 1 个隐藏层
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers += [nn.Linear(d, n_class)]      # 输出层
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = MLP(hidden=64, depth=2).to(device)
print(model)

# ============ 6. 训练配置 ============
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 200
PATIENCE = 30        # 早停：验证 loss 连续 30 轮不降就停

# 记录曲线用
history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}

# 用于保存验证集表现最好的模型
best_val_loss = float('inf')   # 初始为无穷大，任何 loss 都比它小
best_state = None              # 存最佳参数
best_epoch = 0                 # 记录最佳出现在第几轮
wait = 0                       # 早停计数器

def evaluate(model, loader):
    model.eval()
    total_loss, correct, n = 0, 0, 0
    with torch.no_grad():
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
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        tl += loss.item() * len(xb)
        tc += (out.argmax(1) == yb).sum().item()
        tn += len(xb)

    train_loss, train_acc = tl/tn, tc/tn
    val_loss, val_acc = evaluate(model, val_loader)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    # 如果这轮验证 loss 更小，就存下当前参数
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch + 1
        wait = 0                       # 有进步，清零
    else:
        wait += 1                      # 没进步，累加
        if wait >= PATIENCE:
            print(f"早停于第 {epoch+1} 轮（最佳出现在第 {best_epoch} 轮）")
            break

    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1:3d} | train_loss {train_loss:.4f} acc {train_acc:.3f} "
              f"| val_loss {val_loss:.4f} acc {val_acc:.3f}")

# ============ 8. 加载最佳模型，再做测试 ============
model.load_state_dict(best_state)
print(f"\n已加载验证集最佳模型(第 {best_epoch} 轮,val_loss={best_val_loss:.4f})")

test_loss, test_acc = evaluate(model, test_loader)
print(f"测试集：loss {test_loss:.4f}, acc {test_acc:.4f}")

# 保存到磁盘
torch.save(model.state_dict(), 'mlp.pth')

# 加载验证：先重建同样结构的模型
model2 = MLP(hidden=64, depth=2).to(device)
model2.load_state_dict(torch.load('mlp.pth'))
model2.eval()
print("模型加载成功 ✔")

# ============ 9. 画 loss / acc 曲线 ============
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='train')
plt.plot(history['val_loss'],   label='val')
plt.axvline(best_epoch-1, color='r', linestyle='--', label=f'best epoch={best_epoch}')
plt.xlabel('epoch'); plt.ylabel('loss'); plt.title('Loss 曲线'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='train')
plt.plot(history['val_acc'],   label='val')
plt.axvline(best_epoch-1, color='r', linestyle='--', label=f'best epoch={best_epoch}')
plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.title('Accuracy 曲线'); plt.legend()
plt.tight_layout(); plt.show()

# ============ 10. 画决策边界 ============
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