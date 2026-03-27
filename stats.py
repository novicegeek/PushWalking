import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import scipy.stats as stats

file_dir = r"D:\Overseas\German Sport University Cologne\20.Course Materials\TSM11-Project&Applied Research Methods\23.Analysis\Stats"
file_path = os.path.join(file_dir, 'all_subjects_summary_fake置零.csv')
save_dir = os.path.join(file_dir, "Plots")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# --- 参数设置（可根据需要修改） ---
dependent_vars = [
    ('delta_y_post_anchor', 'Max Y Translation after Anchor Step [m]'),
    ('delta_speed_post_anchor_x', 'Max Anteroposterior Speed Change after Anchor Step [m/s]'),
    ('delta_speed_post_anchor_y', 'm/s', 'Max Mediolateral Speed Change after Anchor Step [m/s]'),
    ('hip_response_moment_post_anchor_norm', 'Normalized Hip Response Abduction Moment after Anchor Step [N/kg]'),
    ('knee_response_moment_post_anchor_norm', 'Normalized Knee Response Abduction Moment after Anchor Step [N/kg]'),
    ('ankle_response_moment_post_anchor_norm', 'Normalized Ankle Response Evertion Moment after Anchor Step [N/kg]')
]         # 因变量
speed_var = 'mean_speed_before_anchor' # 自变量1
thrust_var = 'push_impulse_norm'    # 自变量2
subject_var = 'subject_id'             # 分组变量

# 1. 加载数据
df = pd.read_csv(file_path)

# --- 异常值剔除函数 (IQR方法) ---
def exclude_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # 返回剔除异常值后的数据索引
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# 2. 自变量全局清洗 (所有因变量共享)
print("正在清洗自变量异常值...")
df_cleaned = exclude_outliers(df, speed_var)
df_cleaned = exclude_outliers(df_cleaned, thrust_var)

# 3. --- 开始因变量循环 ---
for dv_tuple in dependent_vars:
    dv_col = dv_tuple[0]
    dv_title = dv_tuple[-1] # 总是取最后一个元素作为标题
    
    print(f"\n\n" + "#"*50)
    print(f"正在处理因变量: {dv_title}")

    # 针对当前因变量进一步剔除异常值
    current_data = exclude_outliers(df_cleaned, dv_col).dropna(subset=[dv_col, speed_var, thrust_var, subject_var])
    
    # --- 绘图逻辑 ---
    # 定义统一的标签字典
    label_names = {
        speed_var: "Mean Speed before Anchor Step [m/s]",
        thrust_var: "Normalized Push Impulse [N*s/kg]",
        subject_var: "Subject ID"
    }

    # ==========================================
    # 第一部分：可视化 - 4 个受试者的个体回归线
    # ==========================================
    # 我们分别观察“速度”和“推力”对因变量的影响趋势
    print("正在生成个体趋势图...")

    def custom_plot(x_var, x_label_key, suffix):
        g = sns.lmplot(x=x_var, y=dv_col, hue=subject_var, data=current_data, 
                       aspect=1.2, height=5, scatter_kws={'alpha':0.4},
                       palette='Set1')
        
        # 修改标题和坐标轴
        g.set_axis_labels(label_names[x_label_key], dv_title)
        plt.title(f"Individual Slopes: {dv_col} vs {x_var}")
        
        # 修改图例标题
        g._legend.set_title(label_names[subject_var])
        
        # 自动保存
        file_name = f"Slope_{dv_col}_vs_{suffix}.png"
        g.savefig(os.path.join(save_dir, file_name), dpi=400)
        plt.close() # 释放内存

    # 生成两张图
    custom_plot(speed_var, speed_var, "Speed")
    custom_plot(thrust_var, thrust_var, "Thrust")

    # --- LMM 模型拟合 ---
    formula = f"{dv_col} ~ {speed_var} * {thrust_var}"

    # ==========================================
    # 第二部分：固定斜率模型 (Random Intercept Only)
    # 假设：每个人起始值不同，但对速度/推力的反应剧烈程度相同
    # ==========================================
    print("\n" + "="*30)
    print("正在拟合：固定斜率模型 (Random Intercept Only)")
    try:
        model_fix = smf.mixedlm(formula, current_data, groups=current_data[subject_var])
        res_fix = model_fix.fit()
        print(res_fix.summary())
    except Exception as e:
        print(f"固定斜率模型失败: {e}")

    # ==========================================
    # 第三部分：随机斜率模型 (Random Intercept & Random Slope)
    # 假设：每个人起始值不同，且每个人对速度/推力的反应剧烈程度也不同
    # ==========================================
    print("\n" + "="*30)
    print("正在拟合：随机斜率模型 (Random Intercept & Random Slope)")

    # re_formula 指定哪些变量需要有随机斜率
    # 注意：N=4 时，这个模型极可能报错(不收敛)，如果报错请以固定斜率版为准
    try:
        model_rand = smf.mixedlm(formula, current_data, 
                                 groups=current_data[subject_var], 
                                 re_formula=f"~{speed_var} + {thrust_var}")
        res_rand = model_rand.fit()
        print(res_rand.summary())
        
        # 打印个体斜率
        print("\n个体斜率 (组平均 + 随机效应):")
        for sub_id, ran_eff in res_rand.random_effects.items():
            slope_speed = res_rand.params[speed_var] + ran_eff[speed_var]
            print(f"受试者 {sub_id} 的个体速度斜率: {slope_speed:.4f}")
        for sub_id, ran_eff in res_rand.random_effects.items():
            slope_thrust = res_rand.params[thrust_var] + ran_eff[thrust_var]
            print(f"受试者 {sub_id} 的个体推力斜率: {slope_thrust:.4f}")
        
    except Exception as e:
        print(f"随机斜率模型运行失败或不收敛，原因: {e}")
        print("建议：对于 N=4 的小样本，固定斜率模型通常更稳健。")

    # ==========================================
    # 第四部分：模型后置检验（残差分析）
    # ==========================================
    # 以固定斜率模型结果为例进行检验
    residuals = res_fix.resid

    plt.figure(figsize=(6, 5))
    stats.probplot(residuals, dist="norm", plot=plt)
    
    # 使用你定义的完整标题作为图表标题的一部分
    plt.title(f'Residual Q-Q Plot\n({dv_title})', fontsize=10)
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    
    # 自动保存 Q-Q 图
    qq_file_name = f"Diag_QQ_{dv_col}.png"
    plt.savefig(os.path.join(save_dir, qq_file_name), dpi=400, bbox_inches='tight')
    plt.close() # 必须关闭，否则会叠加到下一张图上

    # 打印正态性检验结果
    shapiro_p = stats.shapiro(residuals)[1]
    print(f"残差正态性检验 P值: {shapiro_p:.4f}")
    if shapiro_p < 0.05:
        print("【注意】残差不符合正态性，建议检查数据或尝试对因变量进行变换。")
    else:
        print("【结论】残差符合正态分布假设。")


print(f"\n所有分析完成，图片已保存至: {save_dir}")