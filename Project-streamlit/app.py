import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ============================================
# Page Config
# ============================================
st.set_page_config(
    page_title="Hotel Booking Predictor",
    page_icon="🏨",
    layout="wide"
)

# ============================================
# Custom CSS - Colorful Professional Design
# ============================================
st.markdown("""
<style>
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }

    /* Top Navigation Bar */
    .nav-bar {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 10px;
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        padding: 15px 30px;
        border-radius: 50px;
        margin-bottom: 30px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .nav-title {
        color: white;
        font-size: 22px;
        font-weight: 700;
        margin-right: 30px;
    }
    .nav-btn {
        background: transparent;
        color: rgba(255,255,255,0.7);
        border: none;
        padding: 8px 20px;
        border-radius: 25px;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.3s;
        text-decoration: none;
    }
    .nav-btn:hover, .nav-btn.active {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .metric-value {
        font-size: 36px;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        color: rgba(255,255,255,0.6);
        font-size: 14px;
        margin-top: 5px;
    }

    /* Section Cards */
    .section-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }

    /* Section Title */
    .section-title {
        color: white;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 15px;
    }

    /* Insight box */
    .insight-box {
        background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
        border-left: 4px solid #667eea;
        border-radius: 10px;
        padding: 15px 20px;
        margin: 10px 0;
        color: rgba(255,255,255,0.85);
        font-size: 15px;
    }

    /* Result boxes */
    .result-cancel {
        background: linear-gradient(135deg, rgba(255,75,75,0.2), rgba(255,0,0,0.1));
        border: 1px solid rgba(255,75,75,0.5);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        color: #ff4b4b;
        font-size: 24px;
        font-weight: 700;
    }
    .result-confirm {
        background: linear-gradient(135deg, rgba(0,255,136,0.2), rgba(0,200,100,0.1));
        border: 1px solid rgba(0,255,136,0.5);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        color: #00ff88;
        font-size: 24px;
        font-weight: 700;
    }

    /* All text white */
    .stMarkdown, p, label, .stSelectbox label,
    .stNumberInput label, .stSlider label {
        color: rgba(255,255,255,0.85) !important;
    }

    /* Input fields */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: white !important;
        border-radius: 10px !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 12px 30px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.3s !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 30px rgba(102,126,234,0.4) !important;
    }

    /* Plot background */
    .stPlotlyChart, .stpyplot {
        border-radius: 15px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Load model and data
# ============================================
@st.cache_resource
def load_model():
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(BASE_DIR, 'hotel_model.pkl'))
    with open(os.path.join(BASE_DIR, 'model_columns.json'), 'r') as f:
        columns = json.load(f)
    return model, columns

@st.cache_data
def load_data():
    df = pd.read_csv('hotel_bookings.csv')
    df['children'] = df['children'].fillna(0)
    df['country'] = df['country'].fillna('Unknown')
    df['agent'] = df['agent'].fillna(0)
    df.drop(columns=['company'], inplace=True)
    df = df[df['adr'] >= 0]
    df = df[df['adr'] <= 5000]
    df = df[df['adults'] > 0]
    df.drop_duplicates(inplace=True)
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['total_guests'] = df['adults'] + df['children'] + df['babies']
    df['arrival_date_month'] = pd.Categorical(
        df['arrival_date_month'],
        categories=['January','February','March','April','May','June',
                    'July','August','September','October','November','December'],
        ordered=True
    )
    return df

model, model_columns = load_model()
df = load_data()

# ============================================
# Plot style
# ============================================
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = '#1a1a2e'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.titlecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = '#333355'
plt.rcParams['grid.color'] = '#2a2a4a'

# ============================================
# Top Navigation
# ============================================
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

col_logo, col1, col2, col3, col4 = st.columns([3, 1, 1, 1, 1])
with col_logo:
    st.markdown("### 🏨 Hotel Booking Predictor")
with col1:
    if st.button("🏠 Home"):
        st.session_state.page = 'Home'
with col2:
    if st.button("📊 EDA"):
        st.session_state.page = 'EDA'
with col3:
    if st.button("🤖 Model"):
        st.session_state.page = 'Model'
with col4:
    if st.button("🔍 Predict"):
        st.session_state.page = 'Predict'

st.markdown("---")
page = st.session_state.page

# ============================================
# PAGE 1: HOME
# ============================================
if page == 'Home':
    st.markdown('<p class="section-title">📋 Project Overview</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("119,390", "Total Bookings"),
        ("37%", "Cancellation Rate"),
        ("32", "Features"),
        ("2015–2017", "Data Period"),
    ]
    for col, (val, label) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="section-card">
        <p class="section-title">🎯 Problem Statement</p>
        Hotels lose significant revenue from unexpected cancellations.
        This project builds a machine learning model to predict whether
        a hotel booking will be cancelled, helping hotels manage
        overbooking and optimize revenue.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="section-card">
        <p class="section-title">📦 Dataset</p>
        <b>Source:</b> Hotel Booking Demand Dataset — Kaggle<br><br>
        <b>Hotels:</b> City Hotel & Resort Hotel (Portugal)<br><br>
        <b>Period:</b> July 2015 — August 2017<br><br>
        <b>Target:</b> <code>is_canceled</code> (0 = Confirmed, 1 = Cancelled)
        </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-title">🗺️ Project Plan</p>', unsafe_allow_html=True)
    steps = [
        ("1️⃣", "Project Understanding", "Define problem & identify target variable"),
        ("2️⃣", "Data Cleaning", "Handle missing values, outliers & duplicates"),
        ("3️⃣", "EDA & Visualization", "Explore 6+ variables with 5+ plot types"),
        ("4️⃣", "Feature Engineering", "Create new features & encode categoricals"),
        ("5️⃣", "Modeling", "Train 3 algorithms & tune best model"),
        ("6️⃣", "Deployment", "Build & deploy this Streamlit web app"),
    ]
    col1, col2 = st.columns(2)
    for i, (icon, title, desc) in enumerate(steps):
        with (col1 if i % 2 == 0 else col2):
            st.markdown(f"""
            <div class="insight-box">
                <b>{icon} {title}</b><br>
                <span style="opacity:0.8">{desc}</span>
            </div>""", unsafe_allow_html=True)

# ============================================
# PAGE 2: EDA
# ============================================
elif page == 'EDA':
    st.markdown('<p class="section-title">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)

    # Plot 1 & 2
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        df['is_canceled'].value_counts().plot(
            kind='bar', color=['#667eea', '#f093fb'], ax=ax, edgecolor='none'
        )
        ax.set_xticklabels(['Not Canceled', 'Canceled'], rotation=0)
        ax.set_title('Overall Cancellation Rate')
        ax.set_ylabel('Number of Bookings')
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('<div class="insight-box">63% of bookings were confirmed, 37% were cancelled</div>',
                    unsafe_allow_html=True)

    with col2:
        fig, ax = plt.subplots(figsize=(5, 3))
        df.groupby('hotel')['is_canceled'].mean().mul(100).plot(
            kind='bar', color=['#667eea', '#f093fb'], ax=ax, edgecolor='none'
        )
        ax.set_title('Cancellation Rate by Hotel Type (%)')
        ax.set_ylabel('Cancellation Rate (%)')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('<div class="insight-box">City Hotel (30%) has higher cancellation than Resort Hotel (24%)</div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Plot 3
    fig, ax = plt.subplots(figsize=(10, 3))
    df['lead_time'].hist(bins=50, color='#667eea', edgecolor='none', ax=ax)
    ax.set_title('Lead Time Distribution')
    ax.set_xlabel('Days Before Arrival')
    ax.set_ylabel('Count')
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown('<div class="insight-box">Most bookings are made within 50 days. Longer lead time = higher cancellation risk</div>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Plot 4
    fig, ax = plt.subplots(figsize=(10, 3))
    df.groupby('arrival_date_month', observed=True).size().plot(
        kind='line', marker='o', color='#f093fb', ax=ax
    )
    ax.set_title('Number of Bookings per Month')
    ax.set_ylabel('Number of Bookings')
    ax.set_xlabel('Month')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown('<div class="insight-box">August is the busiest month. January is the slowest</div>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Plot 5 & 6
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(data=df, x='hotel', y='adr',
                    palette=['#667eea', '#f093fb'], ax=ax)
        ax.set_title('ADR by Hotel Type')
        ax.set_ylabel('Price per Night')
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('<div class="insight-box">City Hotel has higher & more spread out prices</div>',
                    unsafe_allow_html=True)

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        numeric_cols = df.select_dtypes(include='number').drop(
            columns=['arrival_date_year'], errors='ignore'
        )
        sns.heatmap(numeric_cols.corr(), annot=False, cmap='coolwarm',
                    ax=ax, linewidths=0)
        ax.set_title('Correlation Heatmap')
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('<div class="insight-box">lead_time & previous_cancellations most correlated with is_canceled</div>',
                    unsafe_allow_html=True)

# ============================================
# PAGE 3: MODEL RESULTS
# ============================================
elif page == 'Model':
    st.markdown('<p class="section-title">🤖 Model Results</p>', unsafe_allow_html=True)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    for col, (val, label) in zip([col1, col2, col3, col4], [
        ("85.45%", "Accuracy"), ("76.90%", "Precision"),
        ("66.30%", "Recall"),   ("71.21%", "F1 Score")
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label} — Best Model</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Table
    results = pd.DataFrame({
        'Model': ['Logistic Regression', 'Decision Tree',
                  'Random Forest', '🏆 Random Forest (Tuned)'],
        'Accuracy':  [0.7928, 0.8067, 0.8544, 0.8545],
        'Precision': [0.6783, 0.6437, 0.7705, 0.7690],
        'Recall':    [0.4497, 0.6437, 0.6598, 0.6630],
        'F1 Score':  [0.5408, 0.6437, 0.7109, 0.7121]
    })
    st.dataframe(results, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Chart
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(4)
    w = 0.2
    models = ['Logistic\nRegression', 'Decision\nTree',
              'Random\nForest', 'RF\nTuned']
    ax.bar(x - w*1.5, results['Accuracy'],  w, label='Accuracy',  color='#667eea')
    ax.bar(x - w*0.5, results['Precision'], w, label='Precision', color='#f093fb')
    ax.bar(x + w*0.5, results['Recall'],    w, label='Recall',    color='#4facfe')
    ax.bar(x + w*1.5, results['F1 Score'],  w, label='F1 Score',  color='#43e97b')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Comparison')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("<br>", unsafe_allow_html=True)

    # Insights
    st.markdown('<p class="section-title">💡 Key Insights</p>', unsafe_allow_html=True)
    insights = [
        "🏆 Random Forest is the best model with highest Accuracy (85.45%) and F1 Score (71.21%)",
        "📈 Tuning improved Recall from 65.98% to 66.30% — catches more cancellations",
        "✅ All metrics are well above 0.3 — minimum required by the project",
        "🔑 lead_time and previous_cancellations are the most important features",
        "🔑 deposit_type Non Refund strongly indicates cancellation"
    ]
    for insight in insights:
        st.markdown(f'<div class="insight-box">{insight}</div>',
                    unsafe_allow_html=True)

# ============================================
# PAGE 4: PREDICT
# ============================================
elif page == 'Predict':
    st.markdown('<p class="section-title">🔍 Predict Cancellation</p>', unsafe_allow_html=True)
    st.markdown("Fill in the booking details below to get a prediction.")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        hotel = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
        lead_time = st.number_input("Lead Time (days)", 0, 737, 30)
        arrival_month = st.selectbox("Arrival Month", [
            "January","February","March","April","May","June",
            "July","August","September","October","November","December"
        ])
        meal = st.selectbox("Meal Plan", ["BB","FB","HB","SC","Undefined"])
        market_segment = st.selectbox("Market Segment", [
            "Direct","Corporate","Online TA","Offline TA/TO",
            "Complementary","Groups","Undefined","Aviation"
        ])

    with col2:
        deposit_type = st.selectbox("Deposit Type",
            ["No Deposit","Non Refund","Refundable"])
        customer_type = st.selectbox("Customer Type",
            ["Transient","Contract","Transient-Party","Group"])
        adr = st.number_input("Average Daily Rate (ADR)", 0.0, 5000.0, 100.0)
        total_nights = st.number_input("Total Nights", 0, 50, 2)
        total_guests = st.number_input("Total Guests", 1, 55, 2)

    previous_cancellations = st.slider("Previous Cancellations", 0, 26, 0)
    special_requests = st.slider("Total Special Requests", 0, 5, 0)
    booking_changes = st.slider("Booking Changes", 0, 21, 0)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Predict Now"):
        input_dict = {
            'lead_time': lead_time,
            'arrival_date_year': 2025,
            'is_repeated_guest': 0,
            'previous_cancellations': previous_cancellations,
            'previous_bookings_not_canceled': 0,
            'booking_changes': booking_changes,
            'agent': 0,
            'days_in_waiting_list': 0,
            'adr': adr,
            'required_car_parking_spaces': 0,
            'total_of_special_requests': special_requests,
            'total_nights': total_nights,
            'total_guests': total_guests,
            'room_changed': 0,
            'is_returning_guest': 0,
            'total_revenue': adr * total_nights,
            f'hotel_{hotel}': 1,
            f'meal_{meal}': 1,
            f'market_segment_{market_segment}': 1,
            f'deposit_type_{deposit_type}': 1,
            f'customer_type_{customer_type}': 1,
            f'arrival_date_month_{arrival_month}': 1,
        }

        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown("<br>", unsafe_allow_html=True)

        if prediction == 1:
            st.markdown(f"""
            <div class="result-cancel">
                ⚠️ This booking is likely to be CANCELLED<br>
                <span style="font-size:16px;opacity:0.8">
                Cancellation Probability: {probability:.1%}
                </span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-confirm">
                ✅ This booking is likely to be CONFIRMED<br>
                <span style="font-size:16px;opacity:0.8">
                Confirmation Probability: {1-probability:.1%}
                </span>
            </div>""", unsafe_allow_html=True)
