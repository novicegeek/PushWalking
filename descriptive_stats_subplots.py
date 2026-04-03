import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Load path and data ---
file_dir = r"D:\Overseas\German Sport University Cologne\20.Course Materials\TSM11-Project&Applied Research Methods\23.Analysis\Stats" # 请替换为你的实际路径
file_path = os.path.join(file_dir, 'all_subjects_summary_for_stats.csv')
save_dir = os.path.join(file_dir, "Plots")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

df = pd.read_csv(file_path)

# --- 2. Create new rows for pooled data ---
def create_plotting_df(data):
    original = data.copy()
    pooled = data.copy()
    pooled['subject_id'] = 'Pooled'
    return pd.concat([original, pooled], ignore_index=True)

# Capitalize speed and intensity for better aesthetics in plots
df['speed'] = df['speed'].str.capitalize()
df['intensity'] = df['intensity'].str.capitalize()

speed_order = ['Slow', 'Normal', 'Fast']
intensity_order = ['Slight', 'Medium', 'Hard']
subjects = sorted(df['subject_id'].unique().tolist())
hue_order = subjects + ['Pooled']
plot_palette = 'Set1'

# --- 3. Prepare data for subplots ---
df_sub1 = create_plotting_df(df)
df_no_fake = df[df['intensity'] != 'Fake'].copy()
df_sub23 = create_plotting_df(df_no_fake)

# --- 4. Create combined plot ---
# Adjust the width and height to fit the subplots side by side
fig, axes = plt.subplots(1, 3, figsize=(16, 7))

# Common plotting parameters
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

# --- 5. Common legend settings (horizontal arrangement, at the bottom) ---
handles, labels = axes[0].get_legend_handles_labels()

# ncol=len(hue_order) force displaying all legend items in one row, frameon=True adds a border around the legend
fig.legend(
    handles, labels, 
    title='Participant ID', 
    title_fontsize=13,
    loc='lower center', 
    bbox_to_anchor=(0.5, -0.05), # Adjust the vertical position of the legend
    ncol=len(hue_order), 
    frameon=True,
    fontsize=12
)

# Adjust layout to prevent overlap and ensure the legend has enough space
plt.tight_layout(rect=[0, 0.05, 1, 1])

# --- 6. Save figure ---
save_path = os.path.join(save_dir, 'Horizontal_Combined_Plots.png')
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show()

print(f"Horizontal Combined Plots saved to: {save_path}")