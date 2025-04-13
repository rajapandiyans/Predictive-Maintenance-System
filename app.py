import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import mysql.connector
from mysql.connector import Error
import hashlib

# --- DATABASE CONNECTION ---
def create_db_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="pdm"
        )
        return connection
    except Error as e:
        st.error(f"Error connecting to MySQL: {e}")
        return None

# --- INITIALIZE DATABASE TABLES ---
def initialize_database():
    connection = create_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            
            # Create users table if not exists (updated with new fields)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    mobile_number VARCHAR(20),
                    password_hash VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create machine_data table if not exists (updated with sensor descriptions)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS machine_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    vibration_rms FLOAT NOT NULL COMMENT 'Vibration in m/s² (RMS)',
                    temperature FLOAT NOT NULL COMMENT 'Temperature in °C',
                    pressure FLOAT NOT NULL COMMENT 'Pressure in psi',
                    operational_hours FLOAT NOT NULL,
                    rul_prediction FLOAT NOT NULL,
                    maintenance_prediction VARCHAR(50) NOT NULL,
                    anomaly_detection VARCHAR(50) NOT NULL,
                    health_score INT NOT NULL,
                    replacement_days INT NOT NULL,
                    accuracy FLOAT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id INT,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );
            """)
            
            # Create notifications table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notifications (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );
            """)
            
            connection.commit()
            cursor.close()
            connection.close()
        except Error as e:
            st.error(f"Error initializing database: {e}")

# Initialize database tables when the app starts
initialize_database()

# --- HASHING FUNCTION ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- USER REGISTRATION ---
def signup_page():
    with st.popover("🔐 Sign Up"):
        st.title("Create New Account")
        new_username = st.text_input("Username", key="signup_username")
        email = st.text_input("Email", key="signup_email")
        mobile = st.text_input("Mobile Number", key="signup_mobile")
        new_password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm_password")
        
        if st.button("Sign Up"):
            if not new_username.strip() or not email.strip() or not new_password.strip():
                st.error("Username, email and password are required!")
            elif new_password != confirm_password:
                st.error("Passwords do not match!")
            else:
                connection = create_db_connection()
                if connection:
                    try:
                        cursor = connection.cursor()
                        cursor.execute("SELECT username, email FROM users WHERE username = %s OR email = %s;", 
                                       (new_username.strip(), email.strip()))
                        existing_user = cursor.fetchone()
                        
                        if existing_user:
                            if existing_user[0] == new_username.strip():
                                st.error("Username already exists!")
                            else:
                                st.error("Email already exists!")
                        else:
                            password_hash = hash_password(new_password)
                            cursor.execute(
                                "INSERT INTO users (username, email, mobile_number, password_hash) VALUES (%s, %s, %s, %s);",
                                (new_username.strip(), email.strip(), mobile.strip(), password_hash)
                            )
                            connection.commit()
                            st.success("Account created successfully! Please login.")
                            st.balloons()  # Add visual feedback for successful registration
                            st.rerun()
                    except Error as e:
                        st.error(f"Error creating account: {e}")
                    finally:
                        cursor.close()
                        connection.close()

# --- USER AUTHENTICATION ---
def login_page():
    st.title("🔐 Login Page")
    signup_page()
    
    username = st.text_input("Username", placeholder="Enter username")
    password = st.text_input("Password", type="password", placeholder="Enter password")
    
    if st.button("Login"):
        if not username.strip() or not password.strip():
            st.error("Username and password are required!")
        else:
            connection = create_db_connection()
            if connection:
                try:
                    cursor = connection.cursor()
                    cursor.execute(
                        "SELECT id, password_hash FROM users WHERE username = %s;",
                        (username.strip(),)
                    )
                    user_data = cursor.fetchone()
                    
                    if user_data:
                        user_id, stored_hash = user_data
                        if hash_password(password) == stored_hash:
                            st.session_state["logged_in"] = True
                            st.session_state["user_id"] = user_id
                            st.session_state["username"] = username.strip()
                            st.rerun()
                        else:
                            st.error("Invalid Username/Password")
                    else:
                        st.error("Invalid Username/Password")
                except Error as e:
                    st.error(f"Error during login: {e}")
                finally:
                    cursor.close()
                    connection.close()

def logout_button():
    if st.sidebar.button("🔓 Logout"):
        st.session_state.clear()
        st.rerun()

# --- STORE PREDICTION DATA ---
def store_prediction_data(features, prediction, accuracy, user_id):
    features = [0.0 if pd.isna(x) or x is None else float(x) for x in features]
    prediction = {
        'RUL Prediction': 0.0 if pd.isna(prediction['RUL Prediction']) else float(prediction['RUL Prediction']),
        'Maintenance Prediction': str(prediction['Maintenance Prediction']),
        'Anomaly Detection': str(prediction['Anomaly Detection']),
        'Health Score': 0 if pd.isna(prediction['Health Score']) else int(prediction['Health Score']),
        'Replacement Days': 0 if pd.isna(prediction['Replacement Days']) else int(prediction['Replacement Days']),
    }
    accuracy = 0.0 if pd.isna(accuracy) else float(accuracy)
    
    connection = create_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute(
                """INSERT INTO machine_data (
                    vibration_rms, temperature, pressure, operational_hours,
                    rul_prediction, maintenance_prediction, anomaly_detection,
                    health_score, replacement_days, accuracy, user_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);""",
                (
                    features[0], features[1], features[2], features[3],
                    prediction['RUL Prediction'], prediction['Maintenance Prediction'],
                    prediction['Anomaly Detection'], prediction['Health Score'],
                    prediction['Replacement Days'], accuracy, user_id
                )
            )
            connection.commit()
            st.session_state["data_submitted"] = True  # Mark that data has been submitted
            st.success("Prediction data stored successfully!")
        except Error as e:
            st.error(f"Error storing prediction data: {e}")
        finally:
            cursor.close()
            connection.close()

# --- LOAD AND PROCESS DATA ---
def process_data(data):
    # Rename columns to match our specific sensor names
    data = data.rename(columns={
        'sensor_1': 'vibration_rms',
        'sensor_2': 'temperature',
        'sensor_3': 'pressure'
    })
    
    features = ['vibration_rms', 'temperature', 'pressure', 'operational_hours']
    target_rul = 'RUL'
    target_maintenance = 'maintenance'
    scaler = StandardScaler()

    data.fillna(method='ffill', inplace=True)
    data[features] = scaler.fit_transform(data[features])

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        data[features], data[target_rul], test_size=0.2, random_state=42)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        data[features], data[target_maintenance], test_size=0.2, random_state=42)

    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train_reg, y_train_reg)
    reg_accuracy = reg_model.score(X_test_reg, y_test_reg)
    
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_model.fit(X_train_clf, y_train_clf)
    clf_accuracy = clf_model.score(X_test_clf, y_test_clf)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['cluster'] = kmeans.fit_predict(data[features])

    return data, reg_model, clf_model, kmeans, reg_accuracy, clf_accuracy

# --- PREDICT FUNCTION ---
def predict_maintenance(features, reg_model, clf_model, kmeans):
    features = [0.0 if pd.isna(x) or x is None else float(x) for x in features]
    
    rul_pred = reg_model.predict([features])
    maint_pred = clf_model.predict([features])
    cluster_pred = kmeans.predict([features])
    
    anomaly_status = {
        0: 'Normal',
        1: 'Anomaly',
        2: 'Neutral'
    }
    
    health_score = max(0, min(100, int(rul_pred[0] / 500 * 100)))
    replacement_days = int(rul_pred[0] / 24)
    return {
        'RUL Prediction': float(rul_pred[0]),
        'Maintenance Prediction': 'Needs Maintenance' if maint_pred[0] == 1 else 'Normal',
        'Anomaly Detection': anomaly_status.get(cluster_pred[0], 'Unknown'),
        'Health Score': int(health_score),
        'Replacement Days': int(replacement_days),
    }

# --- SEND NOTIFICATIONS ---
def send_notification(message):
    st.warning(f"🔔 Notification Sent: {message}")
    
    if "user_id" in st.session_state:
        connection = create_db_connection()
        if connection:
            try:
                cursor = connection.cursor()
                cursor.execute(
                    "INSERT INTO notifications (user_id, message) VALUES (%s, %s);",
                    (st.session_state["user_id"], message)
                )
                connection.commit()
            except Error as e:
                st.error(f"Error storing notification: {e}")
            finally:
                cursor.close()
                connection.close()

# --- CHECK IF USER HAS DATA ---
def user_has_data(user_id):
    connection = create_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM machine_data WHERE user_id = %s;",
                (user_id,)
            )
            count = cursor.fetchone()[0]
            return count > 0
        except Error as e:
            st.error(f"Error checking user data: {e}")
            return False
        finally:
            cursor.close()
            connection.close()
    return False

# --- MAIN APP ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state.get("logged_in"):
    login_page()
else:
    # Sidebar Menu
    st.sidebar.title(f"🔧 Welcome, {st.session_state.get('username', 'User')}")
    logout_button()

    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["Home", "Upload Data", "Results", "Visualizations"],
            icons=["house", "upload", "check2-circle", "bar-chart-line"],
            menu_icon="cast",
            default_index=0,
        )

    # Default or Uploaded Data
    if "data" not in st.session_state:
        try:
            # Sample data with realistic sensor ranges
            sample_data = pd.DataFrame({
                'vibration_rms': np.random.normal(1.5, 0.5, 100),  # Typical vibration range (m/s²)
                'temperature': np.random.normal(75, 15, 100),      # Typical temperature range (°C)
                'pressure': np.random.normal(100, 20, 100),       # Typical pressure range (psi)
                'operational_hours': np.random.uniform(0, 1000, 100),
                'RUL': np.random.uniform(100, 500, 100),
                'maintenance': np.random.randint(0, 2, 100)
            })
            st.session_state["data"], st.session_state["reg_model"], st.session_state["clf_model"], \
            st.session_state["kmeans"], st.session_state["reg_accuracy"], st.session_state["clf_accuracy"] = process_data(sample_data)
            st.session_state["data_submitted"] = False  # Initialize data submission flag
        except Exception as e:
            st.error(f"Error loading sample data: {e}")

    if selected == "Home":
        st.title("Welcome to the Predictive Maintenance Dashboard")
        st.markdown("""
        ### Machine Health Monitoring System
        This dashboard monitors three critical sensors:
        - **Vibration (RMS in m/s²)**: Measures machine vibration levels (normal range: 0.5-3 m/s²)
        - **Temperature (°C)**: Monitors operating temperature
        - **Pressure (psi)**: Tracks hydraulic/pneumatic pressure
        """)
        if "username" in st.session_state:
            st.write(f"Logged in as: **{st.session_state['username']}**")

    elif selected == "Upload Data":
        st.title("📁 Upload Machine Data")
        
        st.markdown("""
        ### Expected Data Format
        Upload a CSV file with these columns (exact names required):
        - `vibration_rms` - Vibration in m/s² (Root Mean Square)
        - `temperature` - Temperature in °C
        - `pressure` - Pressure in psi
        - `operational_hours` - Total hours of operation
        - `RUL` - Remaining Useful Life (hours)
        - `maintenance` - Binary (0=Normal, 1=Maintenance Needed)
        """)
        
        uploaded_file = st.file_uploader("Upload CSV file:", type=["csv"])
        
        if uploaded_file is not None:
            try:
                uploaded_data = pd.read_csv(uploaded_file)
                required_columns = ['vibration_rms', 'temperature', 'pressure', 
                                  'operational_hours', 'RUL', 'maintenance']
                
                if all(col in uploaded_data.columns for col in required_columns):
                    processed_data, reg_model, clf_model, kmeans, reg_accuracy, clf_accuracy = process_data(uploaded_data)
                    
                    st.write("### Data Preview with Anomaly Detection")
                    display_data = processed_data.copy()
                    display_data['Anomaly Status'] = display_data['cluster'].map({
                        0: 'Normal',
                        1: 'Anomaly',
                        2: 'Neutral'
                    })
                    
                    st.dataframe(display_data[['vibration_rms', 'temperature', 'pressure', 
                                             'operational_hours', 'Anomaly Status']].head(10))
                    
                    # Sensor value statistics
                    st.write("### Sensor Value Ranges")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Vibration (m/s²)", 
                               f"{display_data['vibration_rms'].mean():.2f}",
                               f"Range: {display_data['vibration_rms'].min():.2f}-{display_data['vibration_rms'].max():.2f}")
                    col2.metric("Temperature (°C)", 
                               f"{display_data['temperature'].mean():.1f}",
                               f"Range: {display_data['temperature'].min():.1f}-{display_data['temperature'].max():.1f}")
                    col3.metric("Pressure (psi)", 
                               f"{display_data['pressure'].mean():.1f}",
                               f"Range: {display_data['pressure'].min():.1f}-{display_data['pressure'].max():.1f}")
                    
                    if st.button("Submit Data to Database"):
                        connection = create_db_connection()
                        if connection:
                            try:
                                cursor = connection.cursor()
                                for _, row in processed_data.iterrows():
                                    features = [row['vibration_rms'], row['temperature'], 
                                               row['pressure'], row['operational_hours']]
                                    prediction = predict_maintenance(features, reg_model, clf_model, kmeans)
                                    avg_accuracy = (reg_accuracy + clf_accuracy) / 2 * 100
                                    
                                    cursor.execute(
                                        """INSERT INTO machine_data (
                                            vibration_rms, temperature, pressure, operational_hours,
                                            rul_prediction, maintenance_prediction, anomaly_detection,
                                            health_score, replacement_days, accuracy, user_id
                                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);""",
                                        (
                                            features[0], features[1], features[2], features[3],
                                            prediction['RUL Prediction'], prediction['Maintenance Prediction'],
                                            prediction['Anomaly Detection'], prediction['Health Score'],
                                            prediction['Replacement Days'], avg_accuracy, st.session_state["user_id"]
                                        )
                                    )
                                
                                connection.commit()
                                st.session_state["has_data"] = True
                                st.session_state["data_submitted"] = True  # Mark that data has been submitted
                                st.success(f"Successfully stored {len(processed_data)} records!")
                                
                                st.session_state["data"] = processed_data
                                st.session_state["reg_model"] = reg_model
                                st.session_state["clf_model"] = clf_model
                                st.session_state["kmeans"] = kmeans
                                st.session_state["reg_accuracy"] = reg_accuracy
                                st.session_state["clf_accuracy"] = clf_accuracy
                                
                            except Error as e:
                                st.error(f"Error storing data: {e}")
                            finally:
                                cursor.close()
                                connection.close()
                else:
                    missing_cols = [col for col in required_columns if col not in uploaded_data.columns]
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
            except Exception as e:
                st.error(f"Error processing file: {e}")

    elif selected == "Results":
        st.title("📊 Prediction Results")
        
        if "user_id" in st.session_state:
            # Check both if user has data AND if data has been submitted in this session
            if not st.session_state.get("data_submitted", False):
                st.warning("No prediction data available. Please upload and submit data first.")
                st.stop()
            
            has_data = user_has_data(st.session_state["user_id"])
            
            if not has_data:
                st.warning("No prediction data available. Please upload data first.")
                st.stop()
            
            connection = create_db_connection()
            if connection:
                try:
                    cursor = connection.cursor()
                    cursor.execute(
                        """SELECT vibration_rms, temperature, pressure, operational_hours, 
                           rul_prediction, maintenance_prediction, anomaly_detection,
                           health_score, replacement_days, accuracy 
                           FROM machine_data 
                           WHERE user_id = %s 
                           ORDER BY timestamp DESC 
                           LIMIT 1;""",
                        (st.session_state["user_id"],)
                    )
                    latest_record = cursor.fetchone()
                    
                    if latest_record:
                        st.write("### Latest Sensor Readings")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Vibration (m/s²)", f"{latest_record[0]:.3f}",
                                   help="Normal range: 0.5-3 m/s². Higher values indicate potential issues.")
                        col2.metric("Temperature (°C)", f"{latest_record[1]:.1f}",
                                   help="Normal range: 50-90°C. High temps may indicate overheating.")
                        col3.metric("Pressure (psi)", f"{latest_record[2]:.1f}",
                                   help="Normal range: 80-120 psi. Deviations may signal leaks or blockages.")
                        st.write(f"**Operational Hours:** {latest_record[3]:.1f} hours")
                        
                        st.write("\n### Prediction Results")
                        st.write(f"**Remaining Useful Life (RUL):** {latest_record[4]:.1f} hours")
                        st.write(f"**Maintenance Status:** {latest_record[5]}")
                        st.write(f"**Anomaly Detection:** {latest_record[6]}")
                        st.write(f"**Model Accuracy:** {latest_record[9]:.1f}%")

                        st.subheader("🔋 Machine Health")
                        health_score = latest_record[7]
                        st.progress(health_score / 100)
                        st.write(f"**Health Score:** {health_score}%")
                        
                        if health_score < 30:
                            st.error("⚠️ Immediate attention required! Schedule maintenance now.")
                            send_notification(f"CRITICAL: Machine health is {health_score}%")
                        elif health_score < 60:
                            st.warning("⚠️ Monitor closely. Maintenance may be needed soon.")
                            send_notification(f"WARNING: Machine health is {health_score}%")
                        else:
                            st.success("✅ Machine is operating normally.")
                            send_notification(f"INFO: Machine health is {health_score}%")
                    else:
                        st.warning("No prediction data available. Please upload data first.")
                except Error as e:
                    st.error(f"Error fetching data: {e}")
                finally:
                    cursor.close()
                    connection.close()


    elif selected == "Visualizations":
        st.title("📊 Sensor Data Visualizations")
        st.markdown("""
        ### Understanding Your Machine's Health
        These visualizations help identify trends and potential issues in your sensor data.
        """)
        
        if "user_id" in st.session_state:
            # Check if data has been submitted in this session
            if not st.session_state.get("data_submitted", False):
                st.warning("No data available for visualization. Please upload and submit data first.")
                st.stop()
            
            has_data = user_has_data(st.session_state["user_id"])
            
            if not has_data:
                st.warning("No historical data available. Please upload data first.")
                st.stop()
            
            connection = create_db_connection()
            if connection:
                try:
                    cursor = connection.cursor()
                    cursor.execute(
                        """SELECT vibration_rms, temperature, pressure, operational_hours, 
                           rul_prediction, health_score, timestamp, anomaly_detection
                           FROM machine_data 
                           WHERE user_id = %s 
                           ORDER BY timestamp DESC 
                           LIMIT 100;""",
                        (st.session_state["user_id"],)
                    )
                    data = cursor.fetchall()
                    
                    if data:
                        df = pd.DataFrame(data, columns=[
                            'vibration_rms', 'temperature', 'pressure', 
                            'operational_hours', 'RUL', 'health_score', 
                            'timestamp', 'anomaly_detection'
                        ])
                        
                        # 1. Correlation Heatmap
                        st.subheader("Sensor Correlations")
                        corr_matrix = df[['vibration_rms', 'temperature', 'pressure', 'health_score']].corr()
                        fig1, ax1 = plt.subplots(figsize=(8,6))
                        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax1)
                        st.pyplot(fig1)
                        st.markdown("""
                        **Interpretation:**
                        - Values close to 1 or -1 indicate strong relationships
                        - Helps identify which sensors affect health score most
                        """)
                        
                        # 2. Anomaly Distribution
                        st.subheader("Anomaly Distribution")
                        anomaly_counts = df['anomaly_detection'].value_counts()
                        fig2, ax2 = plt.subplots()
                        ax2.pie(anomaly_counts, labels=anomaly_counts.index, autopct='%1.1f%%', 
                                colors=['green','red','yellow'], startangle=90)
                        ax2.axis('equal')
                        st.pyplot(fig2)
                        st.markdown("""
                        **Anomaly Status:**
                        - Green: Normal operation
                        - Red: Critical issues needing attention
                        - Yellow: Borderline cases to monitor
                        """)
                        
                        # 3. Sensor Trends with Rolling Averages
                        st.subheader("Sensor Trends with 5-Point Averages")
                        df['vibration_rolling'] = df['vibration_rms'].rolling(5).mean()
                        df['temp_rolling'] = df['temperature'].rolling(5).mean()
                        df['pressure_rolling'] = df['pressure'].rolling(5).mean()
                        
                        fig3, (ax3, ax4, ax5) = plt.subplots(3, 1, figsize=(12, 12))
                        
                        # Vibration trend
                        ax3.plot(df['timestamp'], df['vibration_rms'], color='blue', alpha=0.3, label='Raw')
                        ax3.plot(df['timestamp'], df['vibration_rolling'], color='blue', linewidth=2, label='5-Point Avg')
                        ax3.set_title("Vibration Trend")
                        ax3.axhline(y=3, color='r', linestyle='--', label='Danger Threshold')
                        ax3.legend()
                        
                        # Temperature trend
                        ax4.plot(df['timestamp'], df['temperature'], color='orange', alpha=0.3, label='Raw')
                        ax4.plot(df['timestamp'], df['temp_rolling'], color='orange', linewidth=2, label='5-Point Avg')
                        ax4.set_title("Temperature Trend")
                        ax4.axhline(y=90, color='r', linestyle='--', label='Danger Threshold')
                        ax4.legend()
                        
                        # Pressure trend
                        ax5.plot(df['timestamp'], df['pressure'], color='green', alpha=0.3, label='Raw')
                        ax5.plot(df['timestamp'], df['pressure_rolling'], color='green', linewidth=2, label='5-Point Avg')
                        ax5.set_title("Pressure Trend")
                        ax5.axhline(y=120, color='r', linestyle='--', label='Upper Threshold')
                        ax5.axhline(y=80, color='r', linestyle='--', label='Lower Threshold')
                        ax5.legend()
                        
                        plt.tight_layout()
                        st.pyplot(fig3)
                        
                        # 4. Threshold Violations
                        st.subheader("Threshold Violations Count")
                        violations = {
                            'High Vibration': sum(df['vibration_rms'] > 3),
                            'High Temp': sum(df['temperature'] > 90),
                            'Pressure Issues': sum((df['pressure'] < 80) | (df['pressure'] > 120))
                        }
                        fig4, ax6 = plt.subplots()
                        ax6.bar(violations.keys(), violations.values(), color=['red','orange','purple'])
                        plt.xticks(rotation=45)
                        st.pyplot(fig4)
                        
                        # 5. Health Score Analysis
                        st.subheader("Health Score Distribution")
                        fig5, (ax7, ax8) = plt.subplots(1, 2, figsize=(16, 6))
                        
                        # Box plot
                        sns.boxplot(x=df['health_score'], color='lightblue', ax=ax7)
                        ax7.set_title("Health Score Spread")
                        
                        # Scatter plot
                        sns.scatterplot(data=df, x='operational_hours', y='health_score', 
                                        hue='anomaly_detection', palette={'Normal':'green','Anomaly':'red','Neutral':'yellow'}, ax=ax8)
                        ax8.set_title("Health vs Operational Hours")
                        
                        st.pyplot(fig5)
                        st.markdown("""
                        **Health Insights:**
                        - Left plot shows score distribution (median, outliers)
                        - Right plot shows wear patterns over time
                        - Red points indicate critical readings
                        """)
                        
                    else:
                        st.warning("No historical data available. Please upload data first.")
                except Error as e:
                    st.error(f"Error fetching data: {e}")
                finally:
                    cursor.close()
                    connection.close()