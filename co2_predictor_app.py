"""
CO2溶解度预测Streamlit应用
基于物理信息机器学习的CO2在CH4+C2H6混合溶剂中溶解度预测
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="CO2溶解度预测系统",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #F0F8FF;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .solubility-value {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        padding: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'scaler_X' not in st.session_state:
    st.session_state.scaler_X = None
if 'scaler_y' not in st.session_state:
    st.session_state.scaler_y = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'batch_data' not in st.session_state:
    st.session_state.batch_data = None
if 'predictions_df' not in st.session_state:
    st.session_state.predictions_df = None

# ============================================================================
# 辅助函数：格式化数值显示
# ============================================================================

def format_solubility(value):
    """格式化溶解度值为6位小数"""
    if pd.isna(value):
        return "NaN"
    return f"{value:.6f}"

def format_float(value, decimals=6):
    """格式化浮点数为指定小数位数"""
    if pd.isna(value):
        return "NaN"
    return f"{value:.{decimals}f}"

def format_metric(value, decimals=6):
    """格式化指标显示"""
    if pd.isna(value):
        return "NaN"
    return f"{value:.{decimals}f}"

# ============================================================================
# 第1部分：数据预处理函数
# ============================================================================

def load_and_prepare_data(file):
    """加载并准备数据"""
    try:
        # 读取Excel文件
        if file.name.endswith('.xlsx') or file.name.endswith('.xls'):
            df = pd.read_excel(file)
        else:
            # 尝试CSV格式
            df = pd.read_csv(file)

        st.success(f"数据加载成功！数据集形状: {df.shape}")

        # 显示数据基本信息
        with st.expander("查看数据基本信息"):
            st.write("**数据预览:**")
            # 格式化显示，确保数值显示正确
            display_df = df.head().copy()
            for col in display_df.columns:
                if display_df[col].dtype in [np.float64, np.float32]:
                    display_df[col] = display_df[col].apply(lambda x: format_float(x))
            st.dataframe(display_df)

            st.write("**数据统计信息:**")
            stats_df = df.describe().copy()
            # 格式化统计信息
            for col in stats_df.columns:
                if stats_df[col].dtype in [np.float64, np.float32]:
                    stats_df[col] = stats_df[col].apply(lambda x: format_float(x))
            st.dataframe(stats_df)

            st.write("**缺失值统计:**")
            missing_df = pd.DataFrame({
                '列名': df.columns,
                '缺失值数量': df.isnull().sum(),
                '缺失值比例': df.isnull().sum() / len(df) * 100
            })
            st.dataframe(missing_df)

        return df, None

    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        return None, str(e)

def prepare_features(df):
    """准备特征和标签"""
    # 自动识别CO2列
    co2_col = None
    for col in df.columns:
        if 'CO2' in str(col).upper() or '二氧化碳' in str(col):
            co2_col = col
            break

    if co2_col is None:
        # 如果没有找到CO2列，让用户选择
        co2_col = st.selectbox("请选择CO2溶解度列（输出变量）:", df.columns)

    # 特征列：除了CO2列之外的所有列
    feature_cols = [col for col in df.columns if col != co2_col]

    X = df[feature_cols]
    y = df[co2_col]

    return X, y, co2_col, feature_cols

# ============================================================================
# 第2部分：模型训练函数
# ============================================================================

def train_svr_model(X_train, y_train):
    """训练SVR模型"""
    with st.spinner("正在训练SVR模型..."):
        # 数据标准化
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

        # 网格搜索寻找最佳参数
        param_grid = {
            'C': [0.1, 1.0, 10.0, 50.0],
            'gamma': [0.001, 0.01, 0.1, 0.5],
            'epsilon': [0.01, 0.1, 0.2]
        }

        svr = SVR(kernel='rbf')
        grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0)
        grid_search.fit(X_train_scaled, y_train_scaled)

        best_svr = grid_search.best_estimator_

        st.success(f"模型训练完成！最佳参数: {grid_search.best_params_}")
        st.info(f"交叉验证最佳R²: {grid_search.best_score_:.4f}")

        return best_svr, scaler_X, scaler_y

# ============================================================================
# 第3部分：物理约束模型
# ============================================================================

class PhysicsConstrainedSVR:
    """物理约束增强SVR模型"""
    def __init__(self, svr_model, scaler_X, scaler_y, feature_names):
        self.svr_model = svr_model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.feature_names = feature_names

        # 识别特征索引
        self.temp_idx = None
        self.pressure_idx = None
        self.x_prime_idx = None

        for i, name in enumerate(feature_names):
            name_lower = name.lower()
            if 'temp' in name_lower:
                self.temp_idx = i
            elif 'pressure' in name_lower or 'kpa' in name_lower:
                self.pressure_idx = i
            elif 'prime' in name_lower:
                self.x_prime_idx = i

    def predict(self, X, correction_strength=0.1):
        """预测并应用物理约束"""
        # 基础预测
        X_scaled = self.scaler_X.transform(X)
        y_pred_scaled = self.svr_model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        # 应用物理约束
        return self.apply_physics_constraints(X, y_pred, correction_strength)

    def apply_physics_constraints(self, X, y_pred, correction_strength=0.1):
        """应用物理约束"""
        y_corrected = y_pred.copy()

        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        # 获取特征数据
        if self.temp_idx is not None:
            T = X_array[:, self.temp_idx]

        if self.x_prime_idx is not None:
            x_prime = X_array[:, self.x_prime_idx]

        # 1. 单调性约束
        if self.temp_idx is not None and len(T) > 1:
            sorted_idx = np.argsort(T)
            T_sorted = T[sorted_idx]
            y_sorted = y_pred[sorted_idx]

            for i in range(1, len(T_sorted)):
                if T_sorted[i] > T_sorted[i-1] and y_sorted[i] < y_sorted[i-1]:
                    original_idx = sorted_idx[i]
                    y_corrected[original_idx] = y_sorted[i-1] * 0.9 + y_sorted[i] * 0.1

        # 2. 边界条件
        if self.x_prime_idx is not None:
            # 纯甲烷边界
            mask_ch4 = (x_prime < 0.01)
            if np.sum(mask_ch4) > 0 and self.temp_idx is not None:
                T_ch4 = T[mask_ch4]
                x_co2_ch4_true = (-1.68105 + 0.034847*T_ch4 - 2.42185e-4*T_ch4**2 + 5.66534e-7*T_ch4**3) / 100
                y_corrected[mask_ch4] = 0.8 * y_corrected[mask_ch4] + 0.2 * x_co2_ch4_true

            # 纯乙烷边界
            mask_c2h6 = (x_prime > 0.99)
            if np.sum(mask_c2h6) > 0 and self.temp_idx is not None:
                T_c2h6 = T[mask_c2h6]
                x_co2_c2h6_true = (-54.60048 + 0.90685*T_c2h6 - 0.00505*T_c2h6**2 + 9.43932e-6*T_c2h6**3) / 100
                y_corrected[mask_c2h6] = 0.8 * y_corrected[mask_c2h6] + 0.2 * x_co2_c2h6_true

        # 3. 值域约束
        y_corrected = np.clip(y_corrected, 0, 0.5)

        return y_corrected

# ============================================================================
# 第4部分：可视化函数
# ============================================================================

def plot_predictions_comparison(y_true, y_pred_svr, y_pred_physics):
    """绘制预测结果对比图"""
    fig = go.Figure()

    # 添加对角线
    min_val = min(np.min(y_true), np.min(y_pred_svr), np.min(y_pred_physics))
    max_val = max(np.max(y_true), np.max(y_pred_svr), np.max(y_pred_physics))

    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='理想预测线'
    ))

    # 添加SVR预测点
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred_svr,
        mode='markers',
        name='SVR预测',
        marker=dict(color='blue', size=8),
        text=[f"真实值: {format_solubility(y_true[i])}<br>SVR预测: {format_solubility(y_pred_svr[i])}"
              for i in range(len(y_true))],
        hoverinfo='text'
    ))

    # 添加物理约束预测点
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred_physics,
        mode='markers',
        name='物理约束预测',
        marker=dict(color='red', size=8, symbol='x'),
        text=[f"真实值: {format_solubility(y_true[i])}<br>物理约束预测: {format_solubility(y_pred_physics[i])}"
              for i in range(len(y_true))],
        hoverinfo='text'
    ))

    fig.update_layout(
        title='预测结果对比',
        xaxis_title='真实值',
        yaxis_title='预测值',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        hovermode='closest'
    )

    return fig

def plot_residuals_distribution(y_true, y_pred_svr, y_pred_physics):
    """绘制残差分布图"""
    residuals_svr = y_pred_svr - y_true
    residuals_physics = y_pred_physics - y_true

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=residuals_svr,
        name='SVR残差',
        opacity=0.7,
        nbinsx=20,
        marker_color='blue',
        text=[f"残差: {format_solubility(residuals_svr[i])}" for i in range(len(residuals_svr))],
        hoverinfo='text'
    ))

    fig.add_trace(go.Histogram(
        x=residuals_physics,
        name='物理约束残差',
        opacity=0.7,
        nbinsx=20,
        marker_color='red',
        text=[f"残差: {format_solubility(residuals_physics[i])}" for i in range(len(residuals_physics))],
        hoverinfo='text'
    ))

    fig.update_layout(
        title='残差分布对比',
        xaxis_title='残差',
        yaxis_title='频率',
        barmode='overlay',
        height=400,
        hovermode='closest'
    )

    return fig

def plot_feature_importance(feature_names, importance_scores):
    """绘制特征重要性图"""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=importance_scores,
        y=feature_names,
        orientation='h',
        marker_color='lightblue',
        text=[format_float(score, 4) for score in importance_scores],
        textposition='auto'
    ))

    fig.update_layout(
        title='特征重要性分析',
        xaxis_title='重要性得分',
        yaxis_title='特征',
        height=400
    )

    return fig

# ============================================================================
# 第5部分：Streamlit主应用
# ============================================================================

def main():
    # 应用标题
    st.markdown("<h1 class='main-header'>🌡️ CO2溶解度预测系统</h1>", unsafe_allow_html=True)
    st.markdown("### 基于物理信息机器学习(PIML)的CO₂在CH₄+C₂H₆混合溶剂中低温溶解度预测")

    # 创建侧边栏
    with st.sidebar:
        st.markdown("## 🛠️ 模型配置")

        # 数据上传
        st.markdown("### 1. 数据上传")
        uploaded_file = st.file_uploader(
            "上传数据文件 (Excel或CSV)",
            type=['xlsx', 'xls', 'csv']
        )

        if uploaded_file is not None:
            # 加载数据
            df, error = load_and_prepare_data(uploaded_file)

            if df is not None:
                # 准备特征
                X, y, co2_col, feature_names = prepare_features(df)
                st.session_state.feature_names = feature_names

                # 模型训练选项
                st.markdown("### 2. 模型训练")
                if st.button("🚀 开始训练模型", type="primary"):
                    # 分割数据
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=42
                    )

                    # 训练模型
                    svr_model, scaler_X, scaler_y = train_svr_model(X_train, y_train)

                    # 保存到session state
                    st.session_state.trained_model = svr_model
                    st.session_state.scaler_X = scaler_X
                    st.session_state.scaler_y = scaler_y
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test

                    st.success("✅ 模型训练完成并保存！")

        # 预测选项
        st.markdown("### 3. 预测设置")
        correction_strength = st.slider(
            "物理约束强度",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="控制物理约束的强度，0表示无约束，1表示完全约束"
        )

    # 主内容区
    tab1, tab2, tab3, tab4 = st.tabs(["📊 数据探索", "🤖 模型训练", "🔮 单点预测", "📈 批量预测"])

    with tab1:
        st.markdown("<h3 class='sub-header'>数据探索与分析</h3>", unsafe_allow_html=True)

        if uploaded_file is not None and df is not None:
            # 数据概览
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("总样本数", len(df))
            with col2:
                st.metric("特征数量", len(df.columns) - 1)
            with col3:
                st.metric("目标变量", co2_col if 'co2_col' in locals() else "未识别")

            # 数据可视化
            st.markdown("#### 数据分布可视化")
            selected_feature = st.selectbox("选择特征进行可视化:", df.columns)

            if selected_feature != co2_col:
                fig = px.scatter(df, x=selected_feature, y=co2_col,
                               title=f"{selected_feature} vs {co2_col}")
                st.plotly_chart(fig, use_container_width=True)

            # 相关性热力图
            st.markdown("#### 特征相关性热力图")
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                # 格式化相关性矩阵显示
                corr_matrix_formatted = corr_matrix.applymap(lambda x: format_float(x, 4))
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                              title="特征相关性矩阵")
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("<h3 class='sub-header'>模型训练与评估</h3>", unsafe_allow_html=True)

        if st.session_state.trained_model is not None:
            # 模型性能评估
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test

            # 基础预测
            svr_pred = st.session_state.trained_model.predict(
                st.session_state.scaler_X.transform(X_test)
            )
            svr_pred = st.session_state.scaler_y.inverse_transform(
                svr_pred.reshape(-1, 1)
            ).ravel()

            # 物理约束预测
            physics_model = PhysicsConstrainedSVR(
                st.session_state.trained_model,
                st.session_state.scaler_X,
                st.session_state.scaler_y,
                st.session_state.feature_names
            )
            physics_pred = physics_model.predict(X_test, correction_strength)

            # 计算指标
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                svr_r2 = r2_score(y_test, svr_pred)
                st.metric("SVR R²", f"{svr_r2:.6f}")
            with col2:
                physics_r2 = r2_score(y_test, physics_pred)
                st.metric("物理约束 R²", f"{physics_r2:.6f}")
            with col3:
                improvement = physics_r2 - svr_r2
                st.metric("改进量", f"{improvement:+.6f}",
                         delta_color="inverse" if improvement < 0 else "normal")
            with col4:
                svr_mae = mean_absolute_error(y_test, svr_pred)
                st.metric("SVR MAE", f"{svr_mae:.6f}")

            # 可视化
            st.markdown("#### 预测结果对比")
            fig = plot_predictions_comparison(y_test.values, svr_pred, physics_pred)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 残差分布")
            fig2 = plot_residuals_distribution(y_test.values, svr_pred, physics_pred)
            st.plotly_chart(fig2, use_container_width=True)

            # 显示详细预测结果表格
            st.markdown("#### 详细预测结果")
            results_df = pd.DataFrame({
                '真实值': y_test.values,
                'SVR预测值': svr_pred,
                '物理约束预测值': physics_pred,
                'SVR残差': svr_pred - y_test.values,
                '物理约束残差': physics_pred - y_test.values
            })

            # 格式化显示
            display_df = results_df.copy()
            for col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: format_solubility(x))

            st.dataframe(display_df.head(10), use_container_width=True)

            # 下载详细结果
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 下载详细预测结果",
                data=csv,
                file_name="detailed_predictions.csv",
                mime="text/csv"
            )
        else:
            st.info("请先上传数据并训练模型")

    with tab3:
        st.markdown("<h3 class='sub-header'>单点预测</h3>", unsafe_allow_html=True)

        if st.session_state.trained_model is not None and st.session_state.feature_names is not None:
            st.markdown("#### 输入预测参数")

            # 创建输入表单
            input_data = {}
            cols = st.columns(3)

            for i, feature in enumerate(st.session_state.feature_names):
                with cols[i % 3]:
                    # 设置默认值范围
                    if 'temp' in feature.lower():
                        default_val = 180.0
                        min_val = 100.0
                        max_val = 300.0
                    elif 'pressure' in feature.lower():
                        default_val = 1500.0
                        min_val = 100.0
                        max_val = 5000.0
                    elif 'ch4' in feature.lower():
                        default_val = 0.5
                        min_val = 0.0
                        max_val = 1.0
                    elif 'c2h6' in feature.lower():
                        default_val = 0.3
                        min_val = 0.0
                        max_val = 1.0
                    else:
                        default_val = 0.0
                        min_val = -1000.0
                        max_val = 1000.0

                    input_data[feature] = st.number_input(
                        f"{feature}",
                        value=float(default_val),
                        min_value=float(min_val),
                        max_value=float(max_val),
                        step=0.1,
                        format="%.4f"
                    )

            if st.button("🔮 进行预测", type="primary"):
                # 准备输入数据
                input_df = pd.DataFrame([input_data])

                # SVR预测
                svr_pred = st.session_state.trained_model.predict(
                    st.session_state.scaler_X.transform(input_df)
                )
                svr_pred = st.session_state.scaler_y.inverse_transform(
                    svr_pred.reshape(-1, 1)
                )[0][0]

                # 物理约束预测
                physics_model = PhysicsConstrainedSVR(
                    st.session_state.trained_model,
                    st.session_state.scaler_X,
                    st.session_state.scaler_y,
                    st.session_state.feature_names
                )
                physics_pred = physics_model.predict(input_df, correction_strength)[0]

                # 显示结果
                st.markdown("### 🎯 预测结果")

                # 创建美观的结果卡片
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### SVR预测结果")
                    st.markdown(f'<div class="solubility-value">{format_solubility(svr_pred)}</div>',
                               unsafe_allow_html=True)
                    st.markdown(f"**摩尔分数: {format_solubility(svr_pred)}**")

                with col2:
                    st.markdown("##### 物理约束预测结果")
                    st.markdown(f'<div class="solubility-value">{format_solubility(physics_pred)}</div>',
                               unsafe_allow_html=True)
                    st.markdown(f"**摩尔分数: {format_solubility(physics_pred)}**")

                # 显示差异
                col3, col4 = st.columns(2)
                with col3:
                    improvement = physics_pred - svr_pred
                    st.metric("差值", f"{improvement:+.6f}",
                             delta_color="inverse" if improvement < 0 else "normal")

                with col4:
                    improvement_percent = (improvement / svr_pred * 100) if svr_pred != 0 else 0
                    st.metric("相对变化", f"{improvement_percent:+.2f}%",
                             delta_color="inverse" if improvement < 0 else "normal")

                # 显示输入参数
                st.markdown("### 📋 输入参数")
                params_df = pd.DataFrame([input_data]).T
                params_df.columns = ['参数值']
                st.dataframe(params_df, use_container_width=True)
        else:
            st.info("请先训练模型以进行预测")

    with tab4:
        st.markdown("<h3 class='sub-header'>批量预测</h3>", unsafe_allow_html=True)

        if st.session_state.trained_model is not None:
            # 批量数据上传
            st.markdown("#### 上传批量预测数据")
            batch_file = st.file_uploader(
                "上传批量预测文件 (Excel或CSV)",
                type=['xlsx', 'xls', 'csv'],
                key="batch_file"
            )

            if batch_file is not None:
                batch_df, _ = load_and_prepare_data(batch_file)

                if batch_df is not None:
                    # 检查列名是否匹配
                    missing_cols = set(st.session_state.feature_names) - set(batch_df.columns)

                    if len(missing_cols) == 0:
                        # 进行批量预测
                        X_batch = batch_df[st.session_state.feature_names]

                        # SVR预测
                        svr_pred = st.session_state.trained_model.predict(
                            st.session_state.scaler_X.transform(X_batch)
                        )
                        svr_pred = st.session_state.scaler_y.inverse_transform(
                            svr_pred.reshape(-1, 1)
                        ).ravel()

                        # 物理约束预测
                        physics_model = PhysicsConstrainedSVR(
                            st.session_state.trained_model,
                            st.session_state.scaler_X,
                            st.session_state.scaler_y,
                            st.session_state.feature_names
                        )
                        physics_pred = physics_model.predict(X_batch, correction_strength)

                        # 创建结果DataFrame
                        results_df = pd.DataFrame({
                            **batch_df,
                            'SVR预测值': svr_pred,
                            '物理约束预测值': physics_pred,
                            '差值': physics_pred - svr_pred
                        })

                        # 确保所有数值列都保留6位小数
                        for col in ['SVR预测值', '物理约束预测值', '差值']:
                            results_df[col] = results_df[col].round(6)

                        st.session_state.predictions_df = results_df

                        # 显示结果
                        st.markdown("#### 预测结果预览")

                        # 创建格式化显示的数据框
                        display_df = results_df.copy()
                        for col in display_df.columns:
                            if display_df[col].dtype in [np.float64, np.float32]:
                                display_df[col] = display_df[col].apply(lambda x: format_solubility(x))

                        st.dataframe(display_df.head(20), use_container_width=True)

                        # 统计信息
                        st.markdown("#### 📊 预测结果统计")
                        stats_cols = st.columns(4)

                        with stats_cols[0]:
                            st.metric("总样本数", len(results_df))
                        with stats_cols[1]:
                            avg_svr = results_df['SVR预测值'].mean()
                            st.metric("SVR预测均值", format_solubility(avg_svr))
                        with stats_cols[2]:
                            avg_physics = results_df['物理约束预测值'].mean()
                            st.metric("物理约束预测均值", format_solubility(avg_physics))
                        with stats_cols[3]:
                            improvement = avg_physics - avg_svr
                            st.metric("平均改进量", format_solubility(improvement))

                        # 下载按钮
                        st.markdown("#### 💾 下载结果")
                        col1, col2 = st.columns(2)

                        with col1:
                            # CSV格式
                            csv = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 下载CSV格式",
                                data=csv,
                                file_name="co2_solubility_predictions.csv",
                                mime="text/csv"
                            )

                        with col2:
                            # Excel格式
                            output = results_df.to_excel(index=False)
                            st.download_button(
                                label="📥 下载Excel格式",
                                data=output,
                                file_name="co2_solubility_predictions.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

                        # 可视化
                        st.markdown("#### 📈 预测结果分布")
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=results_df['SVR预测值'],
                            name='SVR预测',
                            opacity=0.7,
                            nbinsx=20,
                            marker_color='blue'
                        ))
                        fig.add_trace(go.Histogram(
                            x=results_df['物理约束预测值'],
                            name='物理约束预测',
                            opacity=0.7,
                            nbinsx=20,
                            marker_color='red'
                        ))
                        fig.update_layout(
                            title='预测值分布对比',
                            xaxis_title='CO₂溶解度预测值',
                            yaxis_title='频率',
                            barmode='overlay',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"缺少必要的列: {missing_cols}")
        else:
            st.info("请先训练模型以进行批量预测")

if __name__ == "__main__":
    main()