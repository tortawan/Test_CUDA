import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define file paths
files = {
    '128': r'C:\Users\torta\Desktop\Python\Test_CUDA\Results\200_100_0005_2_128\all_experiments_results.csv',
    '256': r'C:\Users\torta\Desktop\Python\Test_CUDA\Results\200_100_0005_2_256\all_experiments_results.csv',
    '512': r'C:\Users\torta\Desktop\Python\Test_CUDA\Results\200_100_0005_2_512\all_experiments_results.csv'
}

# Read and tag each dataset
dfs = []
for batch, path in files.items():
    df = pd.read_csv(path)
    df['batch_size'] = batch  # add a batch size label for grouping
    dfs.append(df)
df_all = pd.concat(dfs, ignore_index=True)

plt.figure(figsize=(12, 7))
sns.lineplot(
    data=df_all,
    x='episode',
    y='avg_eval_score',
    hue='batch_size',
    estimator='mean',
    ci='sd',               # or ci=95 for 95% confidence interval
    err_style='band'
)
plt.title('Average Evaluation Score by Episode (Batch Size Comparison)')
plt.xlabel('Episode')
plt.ylabel('Average Evaluation Score')
plt.legend(title='Batch Size')
plt.tight_layout()
plt.show()
