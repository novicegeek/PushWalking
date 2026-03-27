import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. 路径与数据加载 ---
# 请确保文件名与您的本地文件一致
file_dir = r"D:\Overseas\German Sport University Cologne\20.Course Materials\TSM11-Project&Applied Research Methods\23.Analysis\Stats"
file_path = os.path.join(file_dir, 'all_subjects_summary_fake置零.csv')
save_dir = os.path.join(file_dir, "Plots")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

df = pd.read_csv(file_path)

# --- 2. 预处理：创建包含 "Pooled" 组的数据集 ---
def create_plotting_df(data):
    # 保留原始数据
    original = data.copy()
    # 创建汇总副本，将所有人的 ID 都改为 "Pooled"
    pooled = data.copy()
    pooled['subject_id'] = 'Pooled'
    # 合并
    return pd.concat([original, pooled], ignore_index=True)

# 定义全局排序和颜色
speed_order = ['slow', 'normal', 'fast']
intensity_order = ['slight', 'medium', 'hard']
# 获取受试者列表并确保 'Pooled' 放在最后
subjects = sorted(df['subject_id'].unique().tolist())
hue_order = subjects + ['Pooled']
# 使用高对比度调色板
plot_palette = 'Set1' 

# --- 3. 绘图函数 ---
def save_boxplot(data, x, y, hue, order, hue_order, xlabel, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=data, x=x, y=y, hue=hue, 
        order=order, hue_order=hue_order, 
        palette=plot_palette,
        showfliers=True, # 显示异常值
        width=0.8
    )
    
    plt.xlabel(xlabel, fontsize=14, labelpad=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, labelpad=12, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # 调整图例
    plt.legend(title='Subject', title_fontsize=13, fontsize=11, 
               bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    
    # 自动调整布局防止标签重叠
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"已保存: {filename}")

# --- 4. 执行绘图 ---

# 图1: Baseline Speed (包含所有数据)
df_plot1 = create_plotting_df(df)
save_boxplot(
    data=df_plot1, 
    x='speed', 
    y='mean_speed_before_anchor', 
    hue='subject_id',
    order=speed_order, 
    hue_order=hue_order,
    xlabel='Block Speed', 
    ylabel='Mean Speed before Anchor Step [m/s]',
    title='Baseline Speed in Different Speed Conditions',
    filename='Boxplot_Baseline_Speed.png'
)

# 排除 'fake' 条件用于图2和图3
df_no_fake = df[df['intensity'] != 'fake'].copy()
df_plot_intensity = create_plotting_df(df_no_fake)

# 图2: Pushing Impulse
save_boxplot(
    data=df_plot_intensity, 
    x='intensity', 
    y='push_impulse_norm', 
    hue='subject_id',
    order=intensity_order, 
    hue_order=hue_order,
    xlabel='Pushing Intensity', 
    ylabel='Normalized Pushing Impulse [N*s/kg]',
    title='Pushing Impulse in Different Pushing Conditions',
    filename='Boxplot_Pushing_Impulse.png'
)

# 图3: Peak Pushing Force
save_boxplot(
    data=df_plot_intensity, 
    x='intensity', 
    y='push_peak_force_norm', 
    hue='subject_id',
    order=intensity_order, 
    hue_order=hue_order,
    xlabel='Pushing Intensity', 
    ylabel='Normalized Peak Pushing Force [N/kg]',
    title='Peak Pushing Force in Different Pushing Conditions',
    filename='Boxplot_Peak_Pushing_Force.png'
)

print(f"\n所有描述性统计图已保存至目录: {save_dir}")