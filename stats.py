import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import scipy.stats as stats

file_dir = r"D:\Overseas\German Sport University Cologne\20.Course Materials\TSM11-Project&Applied Research Methods\23.Analysis\Stats"
file_path = os.path.join(file_dir, 'all_subjects_summary_for_stats.csv')
plot_dir = os.path.join(file_dir, "Plots")
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# --- Settings ---
dependent_vars = [
    ('delta_y_post_anchor', 'Max Y Translation after Anchor Step [m]'),
    ('delta_speed_post_anchor_x', 'Max Anteroposterior Speed Change after Anchor Step [m/s]'),
    ('delta_speed_post_anchor_y', 'm/s', 'Max Mediolateral Speed Change after Anchor Step [m/s]'),
    ('hip_response_moment_post_anchor_norm', 'Normalized Hip Response Abduction Moment after Anchor Step [N/kg]'),
    ('knee_response_moment_post_anchor_norm', 'Normalized Knee Response Abduction Moment after Anchor Step [N/kg]'),
    ('ankle_response_moment_post_anchor_norm', 'Normalized Ankle Response Evertion Moment after Anchor Step [N/kg]')
]
speed_var = 'mean_speed_before_anchor'
thrust_var = 'push_impulse_norm'
subject_var = 'subject_id'

# Results container
results_summary = []

# 1. Load data
df = pd.read_csv(file_path)
# Take the negative of x- and y-axis displacement and velocity change
# to ensure that positive values represent the direction of walking/push
df[dependent_vars[0][0]] *= -1
df[dependent_vars[1][0]] *= -1
df[dependent_vars[2][0]] *= -1

# Helper function: Format statistical results
def format_lmm_res(res, param_name):
    if param_name not in res.params: return "N/A"
    val = res.params[param_name]
    conf = res.conf_int().loc[param_name]
    p = res.pvalues[param_name]
    return f"{val:.3f} ({conf[0]:.3f}, {conf[1]:.3f})\np={p:.4f}"

# --- Exclude Outliers (IQR Method) ---
def exclude_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Return the indices of the non-outlier data points
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# 2. Clean independent variables
print("Excluding outliers of independent variables...")
df_cleaned = exclude_outliers(df, speed_var)
df_cleaned = exclude_outliers(df_cleaned, thrust_var)

# 3. --- Start Dependent Variable Loop ---
for dv_tuple in dependent_vars:
    dv_col = dv_tuple[0]
    dv_title = dv_tuple[-1]
    
    print(f"\n\n" + "#"*50)
    print(f"Processing dependent variable: {dv_title}")

    # Exclude outliers for the current dependent variable
    current_data = exclude_outliers(df_cleaned, dv_col).dropna(subset=[dv_col, speed_var, thrust_var, subject_var])
    # Store the results for the current variable
    row_data = {"Dependent Variable": dv_col, "n": len(current_data)}
    
    # Labels for plots
    label_names = {
        speed_var: "Mean Speed before Anchor Step [m/s]",
        thrust_var: "Normalized Push Impulse [N*s/kg]",
        subject_var: "Participant ID"
    }

    # ==========================================
    # Part 1: Visualization - Individual Regression Lines for 4 Participants
    # ==========================================
    # Observe the trends for speed and pushing impulse separately
    print("Generating individual trend plots...")

    def custom_plot(x_var, x_label_key, suffix):
        g = sns.lmplot(x=x_var, y=dv_col, hue=subject_var, data=current_data, 
                       aspect=1.2, height=5, scatter_kws={'alpha':0.4},
                       palette='Set1')
        
        g.set_axis_labels(label_names[x_label_key], dv_title)
        plt.title(f"Individual Slopes: {dv_col} vs {x_var}")
        
        # Change legend title
        g._legend.set_title(label_names[subject_var])
        
        file_name = f"Slope_{dv_col}_vs_{suffix}.png"
        g.savefig(os.path.join(plot_dir, file_name), dpi=400)
        plt.close() # 释放内存

    # Create separate plots with scatters and regression lines
    custom_plot(speed_var, speed_var, "Speed")
    custom_plot(thrust_var, thrust_var, "Thrust")

    # --- LMM model fitting formula ---
    formula = f"{dv_col} ~ {speed_var} * {thrust_var}"

    # ==========================================
    # Part 2: Fixed Slope Model (Random Intercept Only)
    # Assumption: Each participant has a different starting point, but the same response to speed/push
    # ==========================================
    print("\n" + "="*30)
    print("Fitting Fixed Slope Model (Random Intercept Only)")
    try:
        model_fix = smf.mixedlm(formula, current_data, groups=current_data[subject_var])
        res_fix = model_fix.fit()
        print(res_fix.summary())
        # Extract fixed effects results
        row_data["Intercept"] = format_lmm_res(res_fix, "Intercept")
        row_data["Speed Slope"] = format_lmm_res(res_fix, speed_var)
        row_data["Impulse Slope"] = format_lmm_res(res_fix, thrust_var)
        row_data["Interaction Slope"] = format_lmm_res(res_fix, f"{speed_var}:{thrust_var}")
    except Exception as e:
        print(f"Fixed Slope Model failed: {e}")
    
    # ==========================================
    # Part 3: Post-hoc Tests (Residual Analysis)
    # ==========================================
    # Using the fixed slope model results for testing
    residuals = res_fix.resid

    plt.figure(figsize=(6, 5))
    stats.probplot(residuals, dist="norm", plot=plt)
    
    # Use self-defined title as part of the plot title
    plt.title(f'Residual Q-Q Plot\n({dv_title})', fontsize=10)
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    
    qq_file_name = f"Diag_QQ_{dv_col}.png"
    plt.savefig(os.path.join(plot_dir, qq_file_name), dpi=400, bbox_inches='tight')
    plt.close() # Close the figure to prevent memory issues

    # Print normality test results
    shapiro_p = stats.shapiro(residuals)[1]
    normality = "Yes" if shapiro_p >= 0.05 else "No"
    print(f"Residual Normality Test P-Value: {shapiro_p:.4f}")
    row_data["Residual Normality"] = f"{normality} ({shapiro_p:.4f})"
    if shapiro_p < 0.05:
        print("[Warning] Residuals do NOT follow a normal distribution. Examine the data or consider transformations.")
    else:
        print("[Info] Residuals appear to be normally distributed.")

    # ==========================================
    # Part 4: Random Slope Model (Random Intercept & Random Slope)
    # Assumption: Each participant has a different starting point and a different response to speed/push
    # ==========================================
    print("\n" + "="*30)
    print("Fitting Random Slope Model (Random Intercept & Random Slope)")

    # re_formula designates which variables have random slopes
    # Note: When N=4, this model may fail to converge. If so, please use the fixed slope version.
    try:
        model_rand = smf.mixedlm(formula, current_data, 
                                 groups=current_data[subject_var], 
                                 re_formula=f"~{speed_var} + {thrust_var}")
        res_rand = model_rand.fit()
        print(res_rand.summary())
        
        # Extract individual slopes and sort
        indiv_speeds = []
        indiv_impulses = []
        print("\nIndividual Slopes (Group Mean + Random Effects):")
        for sub_id, ran_eff in res_rand.random_effects.items():
            slope_speed = res_rand.params[speed_var] + ran_eff[speed_var]
            indiv_speeds.append(slope_speed)
            print(f"Participant {sub_id}'s Individual Speed Slope: {slope_speed:.4f}")
        for sub_id, ran_eff in res_rand.random_effects.items():
            slope_thrust = res_rand.params[thrust_var] + ran_eff[thrust_var]
            indiv_impulses.append(slope_thrust)
            print(f"Participant {sub_id}'s Individual Impulse Slope: {slope_thrust:.4f}")
        
        row_data["Indiv Speed Slopes"] = ", ".join([f"{x:.4f}" for x in sorted(indiv_speeds)])
        row_data["Indiv Impulse Slopes"] = ", ".join([f"{x:.4f}" for x in sorted(indiv_impulses)])
        
    except Exception as e:
        print(f"Failed to fit Random Slope Model or it did not converge. Reason: {e}")
        print("Suggestion: For small samples (N=4), the fixed slope model is often more robust.")
        row_data["Indiv Speed Slopes"] = "Model Failed"
        row_data["Indiv Impulse Slopes"] = "Model Failed"
    
    results_summary.append(row_data)

# --- 5. Export Results ---
summary_df = pd.DataFrame(results_summary)
# Adjust column order
cols = ["Dependent Variable", "n", "Intercept", "Speed Slope", "Impulse Slope", 
        "Interaction Slope", "Residual Normality", "Indiv Speed Slopes", "Indiv Impulse Slopes"]
summary_df = summary_df[cols]

summary_path = os.path.join(file_dir, "LMM_Statistical_Summary.xlsx")
summary_df.to_excel(summary_path, index=False)
print(f"\nStatistical summary table saved to: {summary_path}")
print(f"\nAll analyses completed, plots saved to: {plot_dir}")