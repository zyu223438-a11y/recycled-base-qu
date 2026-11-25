from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="基层 7 d 强度预测", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load(Path(__file__).with_name("verified_model.joblib"))

bundle = load_model()
st.title("水泥稳定再生混凝土骨料基层强度预测")
st.caption("XGBoost · 7 d 无侧限抗压强度")
st.subheader("输入参数")
columns = st.columns(3)
groups = [
    ("配合比", [("C", "水泥剂量（%）"), ("R", "再生骨料替代率（%）")]),
    ("再生骨料", [("Ta", "处理方式"), ("Wa", "吸水率（%）"), ("ACV", "压碎值（%）"),
                  ("rho_a", "表观密度（g/cm³）")]),
    ("水泥", [("Tc", "水泥种类"), ("Gc", "水泥强度等级")]),
]
inputs = {}
for column, (title, fields) in zip(columns, groups):
    with column:
        st.markdown(f"**{title}**")
        for key, label in fields:
            default = bundle["defaults"][key]
            if key in bundle["choices"]:
                choices = bundle["choices"][key]
                inputs[key] = st.selectbox(label, choices, index=choices.index(default))
            else:
                lo, hi = bundle["ranges"][key]
                inputs[key] = st.number_input(
                    label, value=float(default), step=0.001 if key == "rho_a" else 0.1,
                    format="%.4f" if key in ("Wa", "ACV", "rho_a") else "%.2f",
                    help=f"数据库范围：{lo:g}～{hi:g}",
                )
                if not lo <= inputs[key] <= hi:
                    st.error(f"请输入 {lo:g}～{hi:g} 范围内的数值。")
if inputs["R"] == 0:
    st.info("零替代率按数据库的对照组口径填写同批备选再生骨料的物性。")
valid = all(lo <= inputs[key] <= hi for key, (lo, hi) in bundle["ranges"].items())
if st.button("预测强度", type="primary", disabled=not valid):
    prediction = bundle["model"].predict(pd.DataFrame([inputs]))[0]
    st.subheader("预测结果")
    st.metric("7 d 无侧限抗压强度 qᵤ", f"{prediction:.3f} MPa")
    st.caption("本次随机留出测试集：41 组，R² = 0.952，RMSE = 0.239 MPa。")
