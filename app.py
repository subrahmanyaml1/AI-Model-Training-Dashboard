import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Classification Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Regression Models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import accuracy_score, r2_score


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="ML Model Trainer Pro", page_icon="🤖", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #0E1117;
}

/* Main Title */
.main-title {
    font-size: 38px;
    font-weight: bold;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}

/* Cards */
.section-card {
    background: white;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 4px 20px rgba(255,255,255,0.08);
    margin-bottom: 20px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: white;
}

/* Uploaded file text */
[data-testid="stFileUploaderFileName"] {
    color: black !important;
}

[data-testid="stFileUploaderFileSize"] {
    color: black !important;
}

/* Drag and drop text */
[data-testid="stFileUploaderDropzone"] {
    color: black !important;
}

[data-testid="stFileUploaderDropzone"] * {
    color: black !important;
}

/* Browse files button */
button[kind="secondary"] {
    color: black !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background-color: #161B22;
    border-radius: 12px;
    padding: 10px;
    box-shadow: 0px 2px 10px rgba(255,255,255,0.05);
}

/* Metric labels and values */
[data-testid="metric-container"] * {
    color: white !important;
}

/* Force metric numeric values white */
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: white !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-size: 16px;
    font-weight: bold;
    color: white !important;
}

/* Active tab */
button[data-baseweb="tab"][aria-selected="true"] {
    color: #00C4B4 !important;
}

/* Buttons */
div.stButton > button {
    background-color: #00C4B4;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: bold;
}

/* Selectbox */
div[data-baseweb="select"] {
    border-radius: 10px;
}

/* Number input */
input {
    border-radius: 10px !important;
}

/* Alert text */
[data-testid="stAlert"] * {
    color: white !important;
}

/* Widget labels */
label {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)


# ---------------- TITLE ----------------
st.markdown('<div class="main-title">🤖 AI Model Training Dashboard</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Control Panel")
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # ---------------- REMOVE FULL NULL ROWS ----------------
    df = df.dropna(how='all')

    # ---------------- REMOVE FULL NULL COLUMNS ----------------
    df = df.dropna(axis=1, how='all')

    # ---------------- FIX OBJECT NUMERIC ----------------
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    original_missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()

    # ---------------- REMOVE DUPLICATES ----------------
    df = df.drop_duplicates()

    # ---------------- TABS ----------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dataset",
        "⚙️ Preprocessing",
        "🤖 Training",
        "🔮 Prediction"
    ])

    # ---------------- TAB 1 ----------------
    with tab1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- TAB 2 ----------------
    with tab2:

        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        drop_cols = st.multiselect("🗑 Select columns to remove manually", df.columns)

        if drop_cols:
            df = df.drop(columns=drop_cols)

        target = st.selectbox("🎯 Select Target Column", df.columns)

        st.dataframe(df.head(10), use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- TARGET ----------------
    X = df.drop(columns=[target])
    y = df[target]

    original_columns = X.columns.tolist()

    # ---------------- REMOVE CONSTANT COLUMNS ----------------
    X = X.loc[:, X.nunique() > 1]

    # ---------------- REMOVE HIGH UNIQUE COLUMNS ----------------
    high_unique_cols = [col for col in X.columns if X[col].nunique() > len(X) * 0.8]

    if high_unique_cols:
        X = X.drop(columns=high_unique_cols)

    # ---------------- PROBLEM TYPE ----------------
    if y.dtype == "object" or y.nunique() <= 20:
        problem_type = "Classification"
    else:
        problem_type = "Regression"

    # ---------------- NULL HANDLING ----------------
    X = X.replace(r'^\s*$', np.nan, regex=True)

    for col in X.columns:

        if X[col].dtype == "object":

            mode_val = X[col].mode()

            if len(mode_val) > 0:
                X[col] = X[col].fillna(mode_val[0])
            else:
                X[col] = X[col].fillna("Unknown")

        else:

            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            X[col] = X[col].fillna(X[col].median())

    # ---------------- TARGET NULL ----------------
    if y.dtype == "object":
        y = y.fillna(y.mode()[0])
    else:
        y = y.fillna(y.median())

    # ---------------- ENCODING ----------------
    categorical_cols = X.select_dtypes(include='object').columns.tolist()

    category_values = {}
    encoders = {}

    for col in categorical_cols:

        category_values[col] = X[col].unique().tolist()

        if X[col].nunique() <= 15:

            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True, dtype=int)
            X = pd.concat([X, dummies], axis=1)
            X = X.drop(columns=[col])

        else:

            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

    # ---------------- TARGET ENCODING ----------------
    if y.dtype == "object":
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)
    else:
        target_encoder = None

    # ---------------- FINAL CLEANING ----------------
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    X = X.astype(float)

    clean_missing = X.isnull().sum().sum()

    # ---------------- DASHBOARD ----------------
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("📄 Rows", df.shape[0])
    col2.metric("🧱 Columns", df.shape[1])
    col3.metric("⚠️ Original Missing", original_missing)
    col4.metric("✅ Clean Missing", clean_missing)
    col5.metric("🗂 Duplicates", duplicates)

    # ---------------- TRAIN TEST ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ---------------- TAB 3 ----------------
    with tab3:

        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        if problem_type == "Classification":
            algo = st.selectbox("Select Algorithm",
                ["KNN", "Naive Bayes", "Decision Tree", "Logistic Regression", "SVM"])
        else:
            algo = st.selectbox("Select Algorithm",
                ["KNN", "Linear Regression", "Decision Tree"])

        if problem_type == "Classification":

            if algo == "KNN":
                model = KNeighborsClassifier()

            elif algo == "Naive Bayes":
                model = GaussianNB()

            elif algo == "Decision Tree":
                model = DecisionTreeClassifier()

            elif algo == "Logistic Regression":
                model = LogisticRegression(max_iter=2000)

            else:
                model = SVC(kernel="linear")

        else:

            if algo == "KNN":
                model = KNeighborsRegressor()

            elif algo == "Linear Regression":
                model = LinearRegression()

            else:
                model = DecisionTreeRegressor()

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        if problem_type == "Classification":
            train_score = accuracy_score(y_train, train_pred)
            test_score = accuracy_score(y_test, test_pred)
        else:
            train_score = r2_score(y_train, train_pred)
            test_score = r2_score(y_test, test_pred)

        st.metric("Train Score", f"{train_score:.4f}")
        st.metric("Test Score", f"{test_score:.4f}")

        diff = abs(train_score - test_score)

        if train_score < 0.70 and test_score < 0.70:
            st.warning("⚠️ Underfitting")
        elif diff <= 0.05:
            st.success("🎯 Best Fit")
        elif diff <= 0.15:
            st.warning("⚠️ Slight Overfitting")
        else:
            st.error("❌ Overfitting")

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- TAB 4 ----------------
    with tab4:

        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        input_dict = {}

        for col in original_columns:

            if col in categorical_cols:

                val = st.selectbox(f"Select {col}", category_values[col])

                dummy_cols = [c for c in X.columns if c.startswith(col + "_")]

                if dummy_cols:

                    for dummy_col in dummy_cols:
                        option = dummy_col.replace(col + "_", "")
                        input_dict[dummy_col] = 1 if val == option else 0

                else:

                    input_dict[col] = encoders[col].transform([val])[0]

            else:

                default_val = float(df[col].median()) if pd.api.types.is_numeric_dtype(df[col]) else 0.0
                input_dict[col] = st.number_input(f"Enter {col}", value=default_val)

        for col in X.columns:
            if col not in input_dict:
                input_dict[col] = 0

        input_df = pd.DataFrame([input_dict])

        input_df = input_df.reindex(columns=X.columns, fill_value=0)

        input_df = input_df.astype(float)

        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)

        st.subheader("Prediction Result")

        if target_encoder is not None:
            prediction = target_encoder.inverse_transform(prediction)

        st.success(prediction[0])

        st.markdown('</div>', unsafe_allow_html=True)
