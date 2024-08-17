import pandas as pd
from sklearn.model_selection import train_test_split
import os

fold = 5
save_dir = '/data115_2/jsh/LEOPARD/splits'
csv_path = '/data115_2/jsh/LEOPARD/leopard_labels/training_labels.csv'

for i in range(fold):
    df = pd.read_csv(csv_path)

    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42 + i)  # 使用不同的随机种子

    train_df = train_df[['case_id']]
    valid_df = valid_df[['case_id']]

    train_df.rename(columns={'case_id': 'train'}, inplace=True)
    valid_df.rename(columns={'case_id': 'val'}, inplace=True)

    combined_df = pd.concat([train_df, valid_df], axis=0)

    save_path = os.path.join(save_dir, 'split'+str(i)+'.csv')
    combined_df.to_csv(save_path, index=False)
 