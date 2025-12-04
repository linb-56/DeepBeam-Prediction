# ==========================================
# app_solid.py: 实腹深受弯构件预测软件 (含个人水印版)
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
        # 请确保这两个pkl文件在同一目录下
        model = joblib.load('solid_model.pkl')
        cols = joblib.load('solid_columns.pkl')
        return model, cols
    except Exception as e:
        return None, None

model, model_cols = load_model()

# --- 2. 侧边栏：参数输入 ---
st.sidebar.header("🛠️ 设计参数输入")

# A. 几何与材料
with st.sidebar.expander("1. 几何与材料", expanded=True):
    b = st.number_input("截面宽度 $b$ (mm)", value=200.0, step=10.0)
    h = st.number_input("截面高度 $h$ (mm)", value=600.0, step=10.0)
    a_h = st.slider("剪跨比 $a/h$", 0.2, 2.5, 1.0, 0.05)
    
    st.markdown("---")
    # === 新增：混凝土类型选择 ===
    agg_option = st.radio(
        "混凝土/骨料类型 (Aggregate)",
        ("普通混凝土 (Normal)", "轻骨料混凝土 (Lightweight)"),
        index=0
    )
    # 逻辑转换：普通=1，轻骨料=2
    aggregate_val = 1 if "普通" in agg_option else 2
    
    fc = st.number_input("混凝土强度 $f_c$ (MPa)", value=30.0, step=5.0)

# B. 配筋信息
st.sidebar.subheader("2. 配筋信息")

# 第一组：纵向钢筋
st.sidebar.markdown("##### 🟢 纵向钢筋")
pl = st.sidebar.number_input("配筋率 $\\rho_l$ (%)", value=1.2, step=0.1, format="%.2f")
fy = st.sidebar.number_input("纵筋屈服强度 $f_y$ (MPa)", value=400.0, step=10.0, format="%.1f")

st.sidebar.markdown("---") 

# 第二组：竖向腹筋
st.sidebar.markdown("##### 🔵 竖向腹筋 (箍筋)")
pv = st.sidebar.number_input("配筋率 $\\rho_v$ (%)", value=0.5, step=0.1, format="%.2f")
fyv = st.sidebar.number_input("箍筋屈服强度 $f_{yv}$ (MPa)", value=300.0, step=10.0, format="%.1f")

st.sidebar.markdown("---") 

# 第三组：水平腹筋
st.sidebar.markdown("##### 🟠 水平腹筋")
ph = st.sidebar.number_input("配筋率 $\\rho_h$ (%)", value=0.5, step=0.1, format="%.2f")
fyh = st.sidebar.number_input("水平筋屈服强度 $f_{yh}$ (MPa)", value=300.0, step=10.0, format="%.1f")


# --- 3. 构造数据 ---
input_dict = {
    'b': b, 
    'h': h, 
    'a/h': a_h, 
    'fc': fc,
    'pl': pl, 
    'fy': fy, 
    'ph': ph, 
    'fyh': fyh, 
    'pv': pv, 
    'fyv': fyv,
    'Aggregate': int(aggregate_val)  # === 新增：加入特征 ===
}

# --- 4. 主界面展示 ---
if model is None:
    st.error("❌ 模型文件丢失！请确保 `solid_model.pkl` 和 `solid_columns.pkl` 在当前目录下。")
else:
    # 数据对齐
    input_df = pd.DataFrame([input_dict])
    final_input = pd.DataFrame()
    
    # 按照训练时的列顺序重排
    missing_cols = []
    for col in model_cols:
        if col in input_df:
            final_input[col] = input_df[col]
        else:
            final_input[col] = 0.0
            missing_cols.append(col)
            
    # 如果有缺失列，在后台打印警告（方便调试）
    if missing_cols:
        print(f"警告：模型需要以下列，但输入中未找到（已自动填0）：{missing_cols}")

    st.title("🧱 实腹深受弯构件承载力预测工具")
    st.markdown("基于 **Stacking 集成学习算法** 开发")
    st.divider()

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.info("### 📝 当前参数概览")
        
        # 显示直观的类型名称
        type_display = "普通混凝土 (Normal)" if aggregate_val == 1 else "轻骨料混凝土 (Lightweight)"
        
        st.markdown(f"""
        * **材料类型**: **{type_display}**
        * **截面尺寸**: ${b:.0f} \\times {h:.0f}$ mm (剪跨比 $a/h={a_h:.2f}$)
        * **混凝土**: $f_c = {fc:.1f}$ MPa
        * **🟢 纵筋**: $\\rho_l = {pl:.2f}\\%$ ($f_y={fy:.0f}$ MPa)
        * **🔵 箍筋**: $\\rho_v = {pv:.2f}\\%$ ($f_{{yv}}={fyv:.0f}$ MPa)
        * **🟠 水平筋**: $\\rho_h = {ph:.2f}\\%$ ($f_{{yh}}={fyh:.0f}$ MPa)
        """)
        
        calc_btn = st.button("🚀 计算承载力", type="primary", use_container_width=True)

    with col2:
        if calc_btn:
            try:
                # 预测
                pred = model.predict(final_input)[0]
                
                st.success("### ✅ 计算完成")
                st.markdown("##### 预测极限受剪承载力 $V_u$")
                # 放大字体显示结果
                st.markdown(f"<h1 style='text-align: left; color: #2e7d32;'>{pred:.2f} kN</h1>", unsafe_allow_html=True)
                
                with st.expander("查看详细数据"):
                    st.write("输入模型的特征矩阵：")
                    st.dataframe(final_input)
            except Exception as e:
                st.error(f"计算出错: {str(e)}")
                st.warning("请检查 `input_dict` 中的键名是否与模型训练时的特征名完全一致。")
        else:
            st.write("👈 请在左侧调整参数并点击计算")

# --- 5. 个性化水印 (Watermark) ---
st.markdown("""
    <style>
    .watermark {
        position: fixed;
        bottom: 10px;
        right: 10px;
        width: auto;
        padding: 5px 10px;
        background-color: rgba(255, 255, 255, 0.7); 
        color: #888888;
        font-size: 14px;
        border-radius: 5px;
        z-index: 9999;
        pointer-events: none;
        font-family: sans-serif;
    }
    @media (prefers-color-scheme: dark) {
        .watermark {
            background-color: rgba(40, 40, 40, 0.7);
            color: #bbbbbb;
        }
    }
    </style>
    
    <div class="watermark">
        © 2025 Developed by Li Yuanxi (Chang'an University) | 毕业设计专用
    </div>
    """, unsafe_allow_html=True)