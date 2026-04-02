import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. 路径与数据加载 ---
file_dir = r"D:\Overseas\German Sport University Cologne\20.Course Materials\TSM11-Project&Applied Research Methods\23.Analysis\Stats" # 请替换为你的实际路径
file_path = os.path.join(file_dir, 'all_subjects_summary_for_stats.csv')
save_dir = os.path.join(file_dir, "Plots")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

df = pd.read_csv(file_path)

# --- 2. 预处理逻辑 ---
def create_plotting_df(data):
    original = data.copy()
    pooled = data.copy()
    pooled['subject_id'] = 'Pooled'
    return pd.concat([original, pooled], ignore_index=True)

# 统一大小写
df['speed'] = df['speed'].str.capitalize()
df['intensity'] = df['intensity'].str.capitalize()

speed_order = ['Slow', 'Normal', 'Fast']
intensity_order = ['Slight', 'Medium', 'Hard']
subjects = sorted(df['subject_id'].unique().tolist())
hue_order = subjects + ['Pooled']
plot_palette = 'Set1'

# --- 3. 准备子图数据 ---
df_sub1 = create_plotting_df(df)
df_no_fake = df[df['intensity'] != 'Fake'].copy()
df_sub23 = create_plotting_df(df_no_fake)

# --- 4. 创建复合图 (1行3列) ---
# 增加宽度 (18)，缩减高度 (7)，确保每个子图看起来很窄
fig, axes = plt.subplots(1, 3, figsize=(16, 7))

# 公共绘图参数
box_params = dict(hue='subject_id', hue_order=hue_order, palette=plot_palette, showfliers=True, width=0.8)

# --- Subplot 1: Baseline Speed ---
sns.boxplot(data=df_sub1, x='speed', y='mean_speed_before_anchor', ax=axes[0], order=speed_order, **box_params)
# axes[0].set_title('A: Baseline Speed', fontsize=15, fontweight='bold', pad=15)
axes[0].set_xlabel('Block Speed', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Baseline Walking Speed [m/s]', fontsize=13, fontweight='bold')
axes[0].get_legend().remove() 

# --- Subplot 2: Peak Pushing Force ---
sns.boxplot(data=df_sub23, x='intensity', y='push_peak_force_norm', ax=axes[1], order=intensity_order, **box_params)
# axes[1].set_title('B: Peak Pushing Force', fontsize=15, fontweight='bold', pad=15)
axes[1].set_xlabel('Pushing Intensity', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Normalized Peak Pushing Force [N/kg]', fontsize=13, fontweight='bold')
axes[1].get_legend().remove()

# --- Subplot 3: Pushing Impulse ---
sns.boxplot(data=df_sub23, x='intensity', y='push_impulse_norm', ax=axes[2], order=intensity_order, **box_params)
# axes[1].set_title('C: Pushing Impulse', fontsize=15, fontweight='bold', pad=15)
axes[2].set_xlabel('Pushing Intensity', fontsize=13, fontweight='bold')
axes[2].set_ylabel('Normalized Pushing Impulse [N·s/kg]', fontsize=13, fontweight='bold')
axes[2].get_legend().remove()

# --- 5. 共享图例设置 (水平排列，位于底部) ---
handles, labels = axes[0].get_legend_handles_labels()

# ncol=len(hue_order) 强制图例在一行显示
fig.legend(
    handles, labels, 
    title='Participant ID', 
    title_fontsize=13,
    loc='lower center', 
    bbox_to_anchor=(0.5, -0.05), # 放在画布最下方
    ncol=len(hue_order), 
    frameon=True,
    fontsize=12
)

# 调整整体间距
# rect=[0, 0.08, 1, 1] 为底部图例留出 8% 的空间
plt.tight_layout(rect=[0, 0.05, 1, 1])

# --- 6. 保存图片 ---
save_path = os.path.join(save_dir, 'Horizontal_Combined_Plots.png')
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show()

print(f"水平复合图已保存至: {save_path}")