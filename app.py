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

# ---------------- CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
}
.main-title {
    font-size: 38px;
    font-weight: bold;
    color: white;
    text-align: center;
}
section[data-testid="stSidebar"] {
    background-color: #111827;
}
section[data-testid="stSidebar"] * {
    color: white;
}
label {
    color: white !important;
}
[data-testid="metric-container"] {
    background-color: #161B22;
    border-radius: 12px;
    padding: 10px;
}
[data-testid="metric-container"] * {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🤖 AI Model Training Dashboard</div>', unsafe_allow_html=True)

# ---------------- CACHE DATA ----------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    df = df.drop_duplicates()
    return df

# ---------------- SIDEBAR ----------------
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = load_data(uploaded_file)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dataset",
        "⚙️ Preprocessing",
        "🤖 Training",
        "🔮 Prediction"
    ])

    # ---------------- TAB 1 ----------------
    with tab1:
        st.dataframe(df.head(10), use_container_width=True)

    # ---------------- TAB 2 ----------------
    with tab2:

        drop_cols = st.multiselect("🗑 Select columns to remove", df.columns)

        df2 = df.copy()

        if drop_cols:
            df2 = df2.drop(columns=drop_cols)

        target = st.selectbox("🎯 Select Target Column", df2.columns)

        st.dataframe(df2.head())

        if st.button("Run Preprocessing"):

            X = df2.drop(columns=[target])
            y = df2[target]

            # Remove constant columns
            X = X.loc[:, X.nunique() > 1]

            # Remove high unique columns
            high_unique_cols = [col for col in X.columns if X[col].nunique() > len(X)*0.8]
            X = X.drop(columns=high_unique_cols)

            # Null handling
            for col in X.columns:
                if X[col].dtype == "object":
                    X[col] = X[col].fillna(X[col].mode()[0])
                else:
                    X[col] = pd.to_numeric(X[col], errors="coerce")
                    X[col] = X[col].fillna(X[col].median())

            # Encoding
            categorical_cols = X.select_dtypes(include='object').columns.tolist()

            encoders = {}

            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le

            # Target encoding
            if y.dtype == "object":
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(y)
            else:
                target_encoder = None

            scaler = StandardScaler()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Store session
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.scaler = scaler
            st.session_state.target_encoder = target_encoder
            st.session_state.encoders = encoders
            st.session_state.target = target
            st.session_state.problem_type = "Classification" if len(np.unique(y)) <= 20 else "Regression"

            st.success("✅ Preprocessing Completed")

    # ---------------- TAB 3 ----------------
    with tab3:

        if "X_train" in st.session_state:

            problem_type = st.session_state.problem_type

            if problem_type == "Classification":
                algo = st.selectbox("Select Algorithm",
                    ["KNN", "Naive Bayes", "Decision Tree", "Logistic Regression", "SVM"])
            else:
                algo = st.selectbox("Select Algorithm",
                    ["KNN", "Linear Regression", "Decision Tree"])

            if st.button("Train Model"):

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
                        model = SVC()

                else:

                    if algo == "KNN":
                        model = KNeighborsRegressor()

                    elif algo == "Linear Regression":
                        model = LinearRegression()

                    else:
                        model = DecisionTreeRegressor()

                model.fit(st.session_state.X_train, st.session_state.y_train)

                train_pred = model.predict(st.session_state.X_train)
                test_pred = model.predict(st.session_state.X_test)

                if problem_type == "Classification":
                    train_score = accuracy_score(st.session_state.y_train, train_pred)
                    test_score = accuracy_score(st.session_state.y_test, test_pred)
                else:
                    train_score = r2_score(st.session_state.y_train, train_pred)
                    test_score = r2_score(st.session_state.y_test, test_pred)

                st.metric("Train Score", f"{train_score:.4f}")
                st.metric("Test Score", f"{test_score:.4f}")

                st.session_state.model = model

    # ---------------- TAB 4 ----------------
    with tab4:

        if "model" in st.session_state:

            input_dict = {}

            for col in st.session_state.X.columns:
                input_dict[col] = st.number_input(f"Enter {col}", value=0.0)

            if st.button("Predict"):

                input_df = pd.DataFrame([input_dict])

                input_scaled = st.session_state.scaler.transform(input_df)

                prediction = st.session_state.model.predict(input_scaled)

                if st.session_state.target_encoder is not None:
                    prediction = st.session_state.target_encoder.inverse_transform(prediction)

                st.success(f"Prediction: {prediction[0]}")
