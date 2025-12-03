# ==========================================
# app_solid.py: 实腹深受弯构件预测软件 (LaTeX格式优化版)
# ==========================================
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 页面设置
st.set_page_config(page_title="实腹深受弯构件计算", layout="wide", page_icon="🧱")

# --- 1. 加载模型 ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('solid_model.pkl')
        cols = joblib.load('solid_columns.pkl')
        return model, cols
    except:
        return None, None

model, model_cols = load_model()

if model is None:
    st.error("❌ 模型加载失败！请先运行第一步代码生成 solid_model.pkl")
    st.stop()

# --- 2. 侧边栏：参数输入 ---
st.sidebar.header("🛠️ 设计参数输入")

# A. 几何与材料
st.sidebar.subheader("1. 几何与材料")
b = st.sidebar.number_input("截面宽度 $b$ (mm)", value=200.0, step=10.0)
h = st.sidebar.number_input("截面高度 $h$ (mm)", value=600.0, step=10.0)
a_h = st.sidebar.slider("剪跨比 $a/h$", 0.2, 2.5, 1.0, 0.05)
fc = st.sidebar.number_input("混凝土强度 $f_c$ (MPa)", value=30.0, step=5.0)

# B. 配筋信息
st.sidebar.subheader("2. 配筋信息")
c1, c2 = st.sidebar.columns(2)

# 左列：纵筋
with c1:
    st.markdown("##### 🟢 纵向钢筋")
    pl = st.sidebar.number_input("配筋率 $\\rho_l$ (%)", value=1.2, step=0.1, format="%.2f")
    fy = st.sidebar.number_input("屈服强度 $f_y$ (MPa)", value=400.0, step=10.0, format="%.1f")

# 右列：腹筋
with c2:
    # 竖向腹筋
    st.markdown("##### 🔵 竖向腹筋 (箍筋)")
    pv = st.sidebar.number_input("配筋率 $\\rho_v$ (%)", value=0.5, step=0.1, format="%.2f")
    fyv = st.sidebar.number_input("屈服强度 $f_{yv}$ (MPa)", value=300.0, step=10.0, format="%.1f")
    
    st.divider() 
    
    # 水平腹筋
    st.markdown("##### 🟠 水平腹筋")
    ph = st.sidebar.number_input("配筋率 $\\rho_h$ (%)", value=0.5, step=0.1, format="%.2f")
    fyh = st.sidebar.number_input("屈服强度 $f_{yh}$ (MPa)", value=300.0, step=10.0, format="%.1f")

# --- 3. 构造数据 ---
# 注意：这里的 key 必须和训练时用的列名完全一致（不含 Latex）
input_dict = {
    'b': b, 'h': h, 'a/h': a_h, 'fc': fc,
    'pl': pl, 'fy': fy, 
    'ph': ph, 'fyh': fyh, 
    'pv': pv, 'fyv': fyv
}

input_df = pd.DataFrame([input_dict])
# 对齐列顺序
final_input = pd.DataFrame()
for col in model_cols:
    final_input[col] = input_df[col] if col in input_df else 0.0

# --- 4. 主界面展示 ---
st.title("🧱 实腹深受弯构件承载力预测工具")
st.markdown("基于 **Stacking 集成学习算法** 开发")
st.divider()

col1, col2 = st.columns([1, 1.5])

with col1:
    st.info("### 📝 当前参数概览")
    # 使用 LaTeX 显示参数摘要
    st.write(f"- **尺寸**: ${b:.0f} \\times {h:.0f}$ mm ($a/h={a_h}$)")
    st.write(f"- **混凝土**: $f_c = {fc}$ MPa")
    st.write(f"- **纵筋**: $\\rho_l = {pl}\\%$ ($f_y={fy}$ MPa)")
    st.write(f"- **箍筋**: $\\rho_v = {pv}\\%$ ($f_{{yv}}={fyv}$ MPa)")
    st.write(f"- **水平筋**: $\\rho_h = {ph}\\%$ ($f_{{yh}}={fyh}$ MPa)")
    
    calc_btn = st.button("🚀 计算承载力", type="primary", use_container_width=True)

with col2:
    if calc_btn:
        # 预测
        pred = model.predict(final_input)[0]
        
        st.success("### ✅ 计算完成")
        # 结果也用 LaTeX
        st.metric(label="极限受剪承载力 ($V_u$)", value=f"{pred:.2f} kN")