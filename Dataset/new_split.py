import pandas as pd
from sklearn.model_selection import train_test_split

# 读取原始CSV文件
csv_path = '/data115_2/jsh/LEOPARD/leopard_labels/training_labels.csv'
df = pd.read_csv(csv_path)

# 按照case_id列进行随机划分
train_val, test = train_test_split(df, test_size=50, random_state=42)  # 先划分出测试集，100张

# 再将剩余的数据划分为训练集和验证集
train, val = train_test_split(train_val, test_size=50, random_state=42)  # 划分验证集，50张

# 提取每个集合的case_id并转换为列表
train_ids = train['case_id'].tolist()
val_ids = val['case_id'].tolist()
test_ids = test['case_id'].tolist()

# 确保每个列表的长度相同
max_length = max(len(train_ids), len(val_ids), len(test_ids))
train_ids += [None] * (max_length - len(train_ids))
val_ids += [None] * (max_length - len(val_ids))
test_ids += [None] * (max_length - len(test_ids))

# 创建一个新的DataFrame来存储划分后的数据
result_df = pd.DataFrame({
    'train': train_ids,
    'val': val_ids,
    'test': test_ids
})

# 将结果保存为新的CSV文件
result_csv_path = '/data115_2/jsh/LEOPARD/leopard_labels/train_val_test_split.csv'
result_df.to_csv(result_csv_path, index=False)

# 打印每个集合的大小
print(f"训练集大小: {len(train)}")
print(f"验证集大小: {len(val)}")
print(f"测试集大小: {len(test)}")

print(f"已保存划分后的数据至 {result_csv_path}")
