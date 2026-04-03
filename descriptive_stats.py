import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Load path and data ---
file_dir = r"D:\Overseas\German Sport University Cologne\20.Course Materials\TSM11-Project&Applied Research Methods\23.Analysis\Stats"
file_path = os.path.join(file_dir, 'all_subjects_summary_for_stats.csv')
save_dir = os.path.join(file_dir, "Plots")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

df = pd.read_csv(file_path)

# --- 2. Preprocessing: Create a dataset containing the "Pooled" group ---
def create_plotting_df(data):
    # Keep the original data
    original = data.copy()
    # Create a pooled copy, changing all subject IDs to "Pooled"
    pooled = data.copy()
    pooled['subject_id'] = 'Pooled'
    # Concatenate
    return pd.concat([original, pooled], ignore_index=True)

# Capitalize speed and intensity for better aesthetics in plots
df['speed'] = df['speed'].str.capitalize()
df['intensity'] = df['intensity'].str.capitalize()

# Define global order and colors
speed_order = ['Slow', 'Normal', 'Fast']
intensity_order = ['Slight', 'Medium', 'Hard']
# Get the list of subjects and ensure 'Pooled' is at the end
subjects = sorted(df['subject_id'].unique().tolist())
hue_order = subjects + ['Pooled']
# Use a high-contrast color palette
plot_palette = 'Set1' 

# --- 3. Plotting Functions ---
# --- Modified plotting function (narrower aspect ratio + legend at the bottom) ---
def save_boxplot(data, x, y, hue, order, hue_order, xlabel, ylabel, title, filename):
    # Adjust figsize: reduce width from 11 to 7, height to 8
    # Current aspect ratio is approximately 0.875 (previously 1.57)
    plt.figure(figsize=(7, 8)) 
    
    ax = sns.boxplot(
        data=data, x=x, y=y, hue=hue, 
        order=order, hue_order=hue_order, 
        palette=plot_palette,
        showfliers=True, 
        width=0.8
    )
    
    plt.xlabel(xlabel.capitalize(), fontsize=14, labelpad=12, fontweight='bold')
    plt.ylabel(ylabel.capitalize(), fontsize=14, labelpad=12, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # --- Modified legend position: move to the bottom of the figure ---
    # loc='upper center' refers to the upper center point of the legend box
    # bbox_to_anchor=(0.5, -0.18) positions that point at 50% horizontal and a certain distance below the coordinate system
    # ncol=3 divides the legend into 3 columns for space efficiency
    plt.legend(
        title='Participant ID', 
        title_fontsize=13, 
        fontsize=11, 
        bbox_to_anchor=(0.5, -0.18), 
        loc='upper center',
        ncol=3, 
        frameon=True
    )
    
    # tight_layout adjusts the padding to prevent overlap
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=400, bbox_inches='tight')  # bbox_inches='tight' ensures the legend is included in the saved figure
    plt.close()
    print(f"Figure saved: {filename}")
# def save_boxplot(data, x, y, hue, order, hue_order, xlabel, ylabel, title, filename):
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(
#         data=data, x=x, y=y, hue=hue, 
#         order=order, hue_order=hue_order, 
#         palette=plot_palette,
#         showfliers=True, # Show outliers
#         width=0.8
#     )
    
#     plt.xlabel(xlabel, fontsize=14, labelpad=12, fontweight='bold')
#     plt.ylabel(ylabel, fontsize=14, labelpad=12, fontweight='bold')
#     plt.title(title, fontsize=16, fontweight='bold', pad=20)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
    
#     # Adjust legend
#     plt.legend(title='Subject', title_fontsize=13, fontsize=11, 
#                bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    
#     # Automatically adjust layout to prevent overlap
#     plt.tight_layout()
    
#     # Save figure
#     save_path = os.path.join(save_dir, filename)
#     plt.savefig(save_path, dpi=400, bbox_inches='tight')
#     plt.close()
#     print(f"Figure saved: {filename}")

# --- 4. Execute Plotting ---

# Fig 1: Baseline Speed (including all data)
df_plot1 = create_plotting_df(df)
save_boxplot(
    data=df_plot1, 
    x='speed', 
    y='mean_speed_before_anchor', 
    hue='subject_id',
    order=speed_order, 
    hue_order=hue_order,
    xlabel='Block Speed', 
    ylabel='Mean Speed before Anchor Strike [m/s]',
    title='Baseline Speed in Different Speed Conditions',
    filename='Boxplot_Baseline_Speed.png'
)

# Exclude 'fake' condition for figures 2 and 3
df_no_fake = df[df['intensity'] != 'fake'].copy()
df_plot_intensity = create_plotting_df(df_no_fake)

# Fig 2: Pushing Impulse
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

# Fig 3: Peak Pushing Force
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

print(f"\nAll descriptive statistics figures saved to directory: {save_dir}")