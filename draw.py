import os
import re
import matplotlib.pyplot as plt

# 日志文件路径，请根据本地实际路径进行替换
file_paths = {
    'origin': 'weights_test/origin_999_train.log',
    'pycbam': 'weights_test/pycbam_895_train.log'
}

# 初始化存储字典
metrics = {
    'origin': {'train_loss': [], 'train_acc': [], 'train_miou': [], 'test_loss': [], 'test_acc': [], 'test_miou': []},
    'pycbam': {'train_loss': [], 'train_acc': [], 'train_miou': [], 'test_loss': [], 'test_acc': [], 'test_miou': []}
}

# 正则表达式
train_pattern = re.compile(r"Train Loss: ([\d.]+) Train acc:([\d.]+)\s+Train mIOU:([\d.]+)")
test_pattern = re.compile(r"Test Loss: ([\d.]+) Test acc:([\d.]+)\s+Test mIOU:([\d.]+)")

# 日志解析函数
def parse_log(path, key):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            train_match = train_pattern.search(line)
            test_match = test_pattern.search(line)
            if train_match:
                metrics[key]['train_loss'].append(float(train_match.group(1)))
                metrics[key]['train_acc'].append(float(train_match.group(2)))
                metrics[key]['train_miou'].append(float(train_match.group(3)))
            elif test_match:
                metrics[key]['test_loss'].append(float(test_match.group(1)))
                metrics[key]['test_acc'].append(float(test_match.group(2)))
                metrics[key]['test_miou'].append(float(test_match.group(3)))

# 执行解析
for key in file_paths:
    parse_log(file_paths[key], key)

# 限制到前600个epoch
max_epoch = 600
for key in metrics:
    for k in metrics[key]:
        metrics[key][k] = metrics[key][k][:max_epoch]

# 创建保存目录
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

# 绘图保存函数
def plot_metric(metric_name, ylabel):
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['origin'][metric_name], label='Origin')
    plt.plot(metrics['pycbam'][metric_name], label='PyCBAM')
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{metric_name.replace('_', ' ').title()} Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{metric_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

# 批量绘图
plot_metric('train_loss', 'Train Loss')
plot_metric('train_acc', 'Train Accuracy')
plot_metric('train_miou', 'Train mIOU')
plot_metric('test_loss', 'Test Loss')
plot_metric('test_acc', 'Test Accuracy')
plot_metric('test_miou', 'Test mIOU')
