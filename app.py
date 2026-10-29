import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone

# ใส่ลิงก์ผลงานของคุณตรงนี้ก่อนเผยแพร่แอป
COLAB_URL = ""
YOUTUBE_URL = ""
POWERPOINT_URL = ""

DATA_URL = "https://raw.githubusercontent.com/abadeanlou/Used-Car-Price-Prediction/main/used_cars.csv"

st.set_page_config(page_title="ระบบทำนายราคารถยนต์มือสอง", page_icon="🚗", layout="wide")
st.markdown("""
<style>
.stApp {background: #10131a; color: #f3f4f6;}
[data-testid="stSidebar"] {background: #171b24;}
.block-container {padding-top: 1.5rem;}
div[data-testid="stMetric"] {background: #1b2230; border: 1px solid #30394a; padding: 14px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.title("🚗 ระบบทำนายราคารถยนต์มือสองด้วย Machine Learning")
st.caption("Used Car Price Prediction · วิเคราะห์ข้อมูลจริง สร้างโมเดล Regression และทดลองทำนายราคา")

@st.cache_data(show_spinner="กำลังโหลดชุดข้อมูล...")
def load_data():
    return pd.read_csv(DATA_URL)

@st.cache_resource(show_spinner="กำลังฝึกโมเดลจากข้อมูล...")
def prepare_models(df, target_col, feature_cols):
    data = df[feature_cols + [target_col]].copy()
    data[target_col] = pd.to_numeric(data[target_col].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    data = data.dropna(subset=[target_col])
    # ทำความสะอาดคอลัมน์ตัวเลขที่เก็บเป็นข้อความ เช่น mileage
    for c in feature_cols:
        if data[c].dtype == "object":
            converted = pd.to_numeric(data[c].astype(str).str.replace(r"[, $]", "", regex=True), errors="coerce")
            if converted.notna().mean() >= 0.75:
                data[c] = converted
    X, y = data[feature_cols], data[target_col]
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2))])
    preprocessor = ColumnTransformer([("num", numeric_pipe, numeric_cols), ("cat", categorical_pipe, categorical_cols)])
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=140, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=140, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Ridge Regression": Ridge(alpha=10.0),
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    fitted, rows = {}, []
    for name, estimator in models.items():
        pipe = Pipeline([("prep", clone(preprocessor)), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        fitted[name] = pipe
        rows.append({"โมเดล": name, "MAE": mean_absolute_error(y_test, pred), "RMSE": mean_squared_error(y_test, pred) ** 0.5, "R²": r2_score(y_test, pred)})
    return data, X_train, X_test, y_train, y_test, fitted, pd.DataFrame(rows).sort_values("RMSE")

with st.sidebar:
    st.header("เมนูระบบ")
    st.write("เลือกแท็บด้านบนเพื่อดูข้อมูล ทดลองทำนาย และเปิดไฟล์ประกอบ")
    st.markdown("---")
    st.caption("ผู้พัฒนา")
    st.write("ชุติพงศ์ ทองคำ")
    st.write("รหัสนักศึกษา: 664245053")
    st.write("หมู่เรียน: 66/44")

try:
    raw_df = load_data()
    data_source = "GitHub – Used Car Price Prediction Dataset (ข้อมูลสาธารณะ)"
except Exception as e:
    st.warning("ไม่สามารถโหลดข้อมูลออนไลน์ได้ กรุณาอัปโหลดไฟล์ CSV ของชุดข้อมูลรถยนต์มือสอง")
    uploaded = st.file_uploader("อัปโหลดไฟล์ CSV", type=["csv"])
    if uploaded is None:
        st.info("เมื่อมีไฟล์ CSV แล้ว ให้อัปโหลดเพื่อเริ่มวิเคราะห์ หรือเชื่อมต่ออินเทอร์เน็ตแล้วรีเฟรชหน้านี้")
        st.stop()
    raw_df = pd.read_csv(uploaded)
    data_source = "ไฟล์ CSV ที่ผู้ใช้เลือกอัปโหลด"

# จับคู่ชื่อคอลัมน์เป้าหมายโดยอัตโนมัติ
lower_map = {str(c).strip().lower().replace(" ", "_"): c for c in raw_df.columns}
target_candidates = ["price", "selling_price", "sellingprice", "sale_price", "listed_price", "msrp"]
target_col = next((lower_map[k] for k in target_candidates if k in lower_map), None)
if target_col is None:
    possible = [c for c in raw_df.columns if "price" in str(c).lower()]
    target_col = possible[0] if possible else None
if target_col is None:
    st.error("ไม่พบคอลัมน์ราคารถในชุดข้อมูล กรุณาตรวจสอบว่าไฟล์มีคอลัมน์ price หรือ selling_price")
    st.write("คอลัมน์ที่พบ:", list(raw_df.columns))
    st.stop()

# ลบคอลัมน์ที่เป็น ID / ข้อความอิสระที่มักมีค่าจำนวนมาก และเลือกฟีเจอร์ที่เหมาะสม
excluded_words = ["id", "url", "image", "description", "listing", "vin", "stock"]
feature_cols = [c for c in raw_df.columns if c != target_col and not any(w in str(c).lower() for w in excluded_words)]
# ลดความซับซ้อน: ไม่ใช้คอลัมน์ที่มีค่าไม่ซ้ำเกิน 90% ของข้อมูล หากเป็นข้อความ
feature_cols = [c for c in feature_cols if not (raw_df[c].dtype == "object" and raw_df[c].nunique(dropna=True) > max(100, 0.9 * len(raw_df)))]
if not feature_cols:
    st.error("ไม่พบคอลัมน์คุณลักษณะสำหรับฝึกโมเดล")
    st.stop()

with st.spinner("กำลังเตรียมข้อมูลและฝึกโมเดลสำหรับการประเมิน..."):
    clean_df, X_train, X_test, y_train, y_test, fitted_models, score_df = prepare_models(raw_df, target_col, feature_cols)

home, explore, predict, compare, files = st.tabs(["หน้าแรก", "สำรวจข้อมูล", "ทดลองทำนายราคา", "เปรียบเทียบโมเดล", "ไฟล์ประกอบ"])

with home:
    st.subheader("ภาพรวมโครงงาน")
    st.write("ระบบนี้ใช้ข้อมูลรายการรถยนต์มือสองจากชุดข้อมูลสาธารณะ เพื่อฝึกโมเดล Machine Learning แบบ Regression สำหรับประมาณราคาขายจากคุณลักษณะของรถ เช่น ปีรถ ระยะทาง ยี่ห้อ และประเภทเชื้อเพลิง (ขึ้นอยู่กับคอลัมน์ที่มีในชุดข้อมูล)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("จำนวนแถวข้อมูล", f"{len(clean_df):,}")
    c2.metric("คุณลักษณะที่ใช้", len(feature_cols))
    c3.metric("ข้อมูลฝึกโมเดล", f"{len(X_train):,}")
    c4.metric("ข้อมูลทดสอบ", f"{len(X_test):,}")
    st.markdown("#### ขั้นตอนการทำงาน")
    st.markdown("1. **Data Collection:** โหลดชุดข้อมูลรถยนต์มือสอง\n2. **Preprocessing:** จัดการค่าที่หายไป แปลงข้อมูลราคา และเข้ารหัสตัวแปรหมวดหมู่\n3. **Model Training:** ฝึกโมเดล Regression หลายแบบ\n4. **Evaluation:** ประเมินด้วย MAE, RMSE และ R²\n5. **Prediction:** ให้ผู้ใช้กรอกคุณลักษณะรถแล้วประเมินราคา")
    st.caption(f"แหล่งข้อมูลที่ใช้: {data_source}")
    st.info("ข้อควรทราบ: ชุดข้อมูลต้นทางเป็นข้อมูลตลาดรถยนต์ต่างประเทศและราคาอาจไม่ได้อยู่ในหน่วยเงินบาท จึงควรใช้เพื่อการศึกษา ไม่ใช่ราคาประเมินซื้อขายจริงในประเทศไทย")

with explore:
    st.subheader("สำรวจข้อมูล (Exploratory Data Analysis)")
    st.write("ตัวอย่างข้อมูลหลังตัดแถวที่ไม่มีราคาออก")
    st.dataframe(clean_df.head(30), use_container_width=True)
    st.download_button("ดาวน์โหลดข้อมูลที่ใช้วิเคราะห์ (CSV)", clean_df.to_csv(index=False).encode("utf-8-sig"), "used_car_cleaned.csv", "text/csv")
    left, right = st.columns(2)
    with left:
        st.markdown("**การกระจายของราคา**")
        fig, ax = plt.subplots()
        ax.hist(clean_df[target_col], bins=30)
        ax.set_xlabel("Price")
        ax.set_ylabel("Number of cars")
        ax.set_title("Price Distribution")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        numeric_df = clean_df.select_dtypes(include=np.number)
        if target_col in numeric_df.columns and len(numeric_df.columns) > 1:
            corr = numeric_df.corr(numeric_only=True)[target_col].drop(target_col, errors="ignore").sort_values()
            st.markdown("**ความสัมพันธ์ของตัวแปรตัวเลขกับราคา**")
            st.bar_chart(corr)
        else:
            st.info("ชุดข้อมูลนี้มีตัวแปรตัวเลขไม่เพียงพอสำหรับคำนวณความสัมพันธ์")

with predict:
    st.subheader("ทดลองทำนายราคารถยนต์มือสอง")
    st.write("กรอกข้อมูลรถด้านล่าง แล้วเลือกรุ่นโมเดลที่ต้องการใช้")
    selected_model = st.selectbox("โมเดล", list(fitted_models.keys()))
    with st.form("prediction_form"):
        vals = {}
        cols = st.columns(2)
        for i, c in enumerate(feature_cols):
            series = raw_df[c]
            with cols[i % 2]:
                if pd.api.types.is_numeric_dtype(series):
                    median = pd.to_numeric(series, errors="coerce").median()
                    if pd.isna(median): median = 0.0
                    min_v = pd.to_numeric(series, errors="coerce").min()
                    max_v = pd.to_numeric(series, errors="coerce").max()
                    if pd.isna(min_v) or pd.isna(max_v) or min_v == max_v:
                        min_v, max_v = float(median - 1), float(median + 1)
                    vals[c] = st.number_input(str(c).replace("_", " ").title(), min_value=float(min_v), max_value=float(max_v), value=float(np.clip(median, min_v, max_v)))
                else:
                    opts = series.dropna().astype(str).value_counts().head(80).index.tolist()
                    if not opts: opts = ["Unknown"]
                    vals[c] = st.selectbox(str(c).replace("_", " ").title(), opts)
        submitted = st.form_submit_button("ทำนายราคา")
    if submitted:
        input_df = pd.DataFrame([vals], columns=feature_cols)
        prediction = float(fitted_models[selected_model].predict(input_df)[0])
        st.success(f"ราคาที่โมเดลประเมิน: {prediction:,.2f} (หน่วยตามชุดข้อมูลต้นทาง)")
        st.caption("ผลลัพธ์เป็นค่าประมาณจากข้อมูลฝึกสอน ไม่ใช่การรับประกันราคาขายจริง")

with compare:
    st.subheader("เปรียบเทียบประสิทธิภาพโมเดล")
    st.write("โมเดลทุกตัวใช้ชุดฝึกและชุดทดสอบเดียวกันเพื่อให้เปรียบเทียบได้อย่างเป็นธรรม")
    st.dataframe(score_df.style.format({"MAE": "{:,.2f}", "RMSE": "{:,.2f}", "R²": "{:.3f}"}), use_container_width=True)
    best = score_df.iloc[0]
    st.success(f"โมเดลที่มีค่า RMSE ต่ำที่สุดในรอบนี้คือ **{best['โมเดล']}** (RMSE = {best['RMSE']:,.2f})")
    st.caption("MAE และ RMSE ยิ่งต่ำโดยทั่วไปยิ่งดี ส่วน R² ยิ่งใกล้ 1 ยิ่งอธิบายความแปรปรวนของราคาได้ดี ทั้งนี้ควรพิจารณาหลายค่า ไม่ใช่ค่าเดียว")
    st.bar_chart(score_df.set_index("โมเดล")[["MAE", "RMSE"]])

with files:
    st.subheader("📎 ไฟล์ประกอบ: Colab / YouTube / PowerPoint")
    st.write("รวมลิงก์สำหรับให้อาจารย์เปิดดูโค้ด วิดีโอนำเสนอ และสไลด์ประกอบโครงงาน")
    link_items = [("Google Colab", COLAB_URL, "เปิด Notebook สำหรับโค้ดและการทดลอง"), ("YouTube", YOUTUBE_URL, "เปิดวิดีโอสาธิตหรือการนำเสนอ"), ("PowerPoint", POWERPOINT_URL, "เปิดสไลด์นำเสนอผลงาน")]
    for label, url, description in link_items:
        st.markdown(f"**{label}** — {description}")
        if url.strip():
            st.markdown(f"[คลิกเพื่อเปิด {label}]({url})")
        else:
            st.caption("ยังไม่ได้ใส่ลิงก์ — เพิ่ม URL ในตัวแปรด้านบนของไฟล์ app.py แล้วอัปโหลดขึ้น GitHub ใหม่")
        st.markdown("---")
    st.markdown("**วิธีเพิ่มลิงก์:** เปิดไฟล์ `app.py` แล้วแก้ค่า `COLAB_URL`, `YOUTUBE_URL`, `POWERPOINT_URL` ที่ต้นไฟล์ จากนั้นบันทึกและอัปโหลดไฟล์ที่แก้แล้วขึ้น GitHub")
    st.markdown("**แหล่งข้อมูล Dataset:** [Used Car Price Prediction Dataset บน GitHub](https://github.com/abadeanlou/Used-Car-Price-Prediction)")

st.markdown("---")
st.caption("โครงงานเพื่อการศึกษา | Machine Learning for Used Car Price Prediction")
