import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

st.set_page_config(page_title="Dashboard Klasifikasi Pendeteksi Penipuan", layout="wide")

# st.title("🎯 Dashboard Klasifikasi Deteksi Penipuan Transaksi")

# Load Models
@st.cache_resource
def load_assets():
    try:
        scaler = joblib.load('scaler.pkl')
        encoder = joblib.load('encoder.pkl')
        numerical_cols = joblib.load('numerical_cols.pkl')
        categorical_cols = joblib.load('categorical_cols.pkl')
        model = joblib.load('model.pkl')
        y_test = joblib.load("y_Test.pkl")
        X_test = joblib.load("X_Test.pkl")
        return scaler, encoder, numerical_cols, categorical_cols, model, y_test, X_test
    except FileNotFoundError as e:
        st.error(f"File model atau preprocessing tidak ditemukan: {e}")
        st.stop()

scaler, encoder, numerical_cols, categorical_cols, model, y_test, X_test = load_assets()

# Feature Engineering
def categorize_part_of_day(hour):
    if 0 <= hour < 6:
        return 'dinihari'
    elif 6 <= hour < 12:
        return 'pagi'
    elif 12 <= hour < 18:
        return 'siang'
    else:
        return 'malam'

# Preprocess Input
def preprocess_input(data):
    df_input = pd.DataFrame([data])

    # Konversi waktu
    df_input['Timestamp'] = pd.to_datetime(df_input['Timestamp'], errors='coerce')
    df_input['hour'] = df_input['Timestamp'].dt.hour
    df_input['day'] = df_input['Timestamp'].dt.day
    df_input['dayofweek'] = df_input['Timestamp'].dt.dayofweek
    df_input['is_night'] = df_input['hour'].apply(lambda x: 1 if x < 5 or x >= 22 else 0)
    df_input['part_of_day'] = df_input['hour'].apply(categorize_part_of_day)
    df_input.drop(columns=['Timestamp'], inplace=True)

    # Scaling
    try:
        df_input[numerical_cols] = df_input[numerical_cols].astype(float)
        df_input[numerical_cols] = scaler.transform(df_input[numerical_cols])
    except Exception as e:
        st.error(f"Kesalahan saat scaling data numerik: {e}")
        st.stop()

    # Encoding
    # Isi NaN dan konversi ke string untuk kolom kategorikal
    for col in categorical_cols:
        df_input[col] = df_input[col].fillna("Unknown").astype(str)

    try:
        encoded = encoder.transform(df_input[categorical_cols])
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan encoding: {e}")
        st.stop()

    # Tambahkan kolom encoded
    for i, col in enumerate(categorical_cols):
        df_input[col + "_encoded"] = encoded[:, i]
    df_input.drop(columns=categorical_cols, inplace=True)

    # Tambahkan kolom kosong jika ada yang hilang
    for col in model.feature_names_in_:
        if col not in df_input.columns:
            df_input[col] = 0

    df_input = df_input[model.feature_names_in_]
    return df_input

# Dataset Overview
def dataset():
    st.title("Informasi Dataset")

    # Deskripsi Dataset
    st.subheader("Deskripsi Dataset")

    st.write("""
        Kumpulan data ini dirancang untuk membantu para praktisi data science, dan peneliti dalam mengembangkan 
        serta mengevaluasi model deteksi penipuan transaksi (fraud detection) yang akurat dan andal. Dataset ini berisi 
        data transaksi sintetis yang disusun menyerupai kondisi transaksi nyata, dengan mencakup berbagai informasi, seperti 
        karakteristik pengguna, jenis transaksi, nominal transaksi, metode pembayaran, serta skor risiko yang dapat digunakan 
        untuk mengidentifikasi potensi aktivitas penipuan. Setiap data memiliki sejumlah fitur yang menggambarkan karakteristik transaksi 
        dan label yang menunjukkan apakah suatu transaksi tergolong penipuan atau bukan. Dengan karakteristik tersebut, 
        dataset ini dapat digunakan untuk memahami pola transaksi, menganalisis hubungan antarfitur, serta melatih dan 
        menguji model machine learning dalam menyelesaikan permasalahan klasifikasi biner (binary classification).
    """)

    st.divider()

    st.subheader("Dimensi Data")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Jumlah Baris", value="50,000")

    with col2:
        st.metric(label="Jumlah Kolom", value="21")
    
    st.metric(label="Fitur Dataset", value="")
    
    feature_data = [
        {
            "Nama Fitur": "Transaction_ID",
            "Deskripsi": "Identitas unik untuk setiap transaksi."
        },
        {
            "Nama Fitur": "User_ID",
            "Deskripsi": "Identitas unik untuk setiap pengguna."
        },
        {
            "Nama Fitur": "Transaction_Amount",
            "Deskripsi": "Jumlah uang yang terlibat dalam transaksi."
        },
        {
            "Nama Fitur": "Transaction_Type",
            "Deskripsi": "Jenis transaksi, seperti online, in-store, atau ATM."
        },
        {
            "Nama Fitur": "Timestamp",
            "Deskripsi": "Tanggal dan waktu terjadinya transaksi."
        },
        {
            "Nama Fitur": "Account_Balance",
            "Deskripsi": "Saldo rekening pengguna sebelum transaksi dilakukan."
        },
        {
            "Nama Fitur": "Device_Type",
            "Deskripsi": "Jenis perangkat yang digunakan, seperti mobile atau desktop."
        },
        {
            "Nama Fitur": "Location",
            "Deskripsi": "Lokasi geografis tempat transaksi dilakukan."
        },
        {
            "Nama Fitur": "Merchant_Category",
            "Deskripsi": "Kategori merchant atau pedagang, seperti retail, food, atau travel."
        },
        {
            "Nama Fitur": "IP_Address_Flag",
            "Deskripsi": "Indikator apakah alamat IP ditandai sebagai mencurigakan (0 atau 1)."
        },
        {
            "Nama Fitur": "Previous_Fraudulent_Activity",
            "Deskripsi": "Jumlah aktivitas penipuan yang pernah dilakukan oleh pengguna sebelumnya."
        },
        {
            "Nama Fitur": "Daily_Transaction_Count",
            "Deskripsi": "Jumlah transaksi yang dilakukan pengguna dalam satu hari."
        },
        {
            "Nama Fitur": "Avg_Transaction_Amount_7d",
            "Deskripsi": "Rata-rata nominal transaksi pengguna selama 7 hari terakhir."
        },
        {
            "Nama Fitur": "Failed_Transaction_Count_7d",
            "Deskripsi": "Jumlah transaksi gagal selama 7 hari terakhir."
        },
        {
            "Nama Fitur": "Card_Type",
            "Deskripsi": "Jenis kartu pembayaran yang digunakan, seperti credit, debit, atau prepaid."
        },
        {
            "Nama Fitur": "Card_Age",
            "Deskripsi": "Usia kartu dalam satuan bulan."
        },
        {
            "Nama Fitur": "Transaction_Distance",
            "Deskripsi": "Jarak antara lokasi yang biasa digunakan pengguna dan lokasi transaksi."
        },
        {
            "Nama Fitur": "Authentication_Method",
            "Deskripsi": "Metode autentikasi yang digunakan, seperti PIN atau biometrik."
        },
        {
            "Nama Fitur": "Risk_Score",
            "Deskripsi": "Skor risiko penipuan yang dihitung untuk transaksi."
        },
        {
            "Nama Fitur": "Is_Weekend",
            "Deskripsi": "Indikator apakah transaksi terjadi pada akhir pekan (0 atau 1)."
        },
        {
            "Nama Fitur": "Fraud_Label",
            "Deskripsi": "Label target klasifikasi, yaitu 0 untuk transaksi normal dan 1 untuk transaksi penipuan."
        }
    ]


    feature_df = pd.DataFrame(feature_data)
    
    st.dataframe(
            feature_df,
            use_container_width=True,
            hide_index=True
        )

# Algorithm Explanation
def algorithm():
    st.title("🌳 Random Forest Algorithm")

    st.divider()

    # 1. Pengertian
    st.subheader("Pengertian Random Forest")

    st.write("""
        Random Forest merupakan algoritma supervised machine learning yang menggabungkan sejumlah decision tree untuk menghasilkan
        prediksi yang lebih stabil. Dalam proyek ini, Random Forest digunakan untuk melakukan klasifikasi transaksi 
        ke dalam dua kelas, yaitu fraud dan non-fraud, berdasarkan karakteristik transaksi yang diberikan.
    """)

    st.divider()

    # 2. Cara Kerja
    st.subheader("Cara Kerja Algoritma")

    steps = [
        (
            "1. Pengambilan Sampel Data",
            "Model membangun pohon keputusan menggunakan sampel data pelatihan yang diambil "
            "melalui bootstrap sampling yaitu pengambilan sampel secara acak dengan pengembalian "
            "(sampling with replacement). Setiap pohon keputusan memperoleh sampel yang dapat berbeda dari pohon lainnya."
        ),
        (
            "2. Pembentukan Decision Tree",
            "Setiap pohon keputusan menentukan pemisahan data berdasarkan fitur yang " 
            " tersedia. Pada setiap percabangan, Random Forest memilih secara acak sebagian "
            "fitur sebagai kandidat pemisahan, kemudian menentukan pemisahan terbaik dari "
            " kandidat tersebut berdasarkan kriteria tertentu"
        ),
        (
            "3. Penggabungan Prediksi",
            "Setiap pohon menghasilkan prediksi berdasarkan pola yang dipelajarinya. "
            "Untuk klasifikasi, Random Forest umumnya menggabungkan prediksi melalui "
            "majority voting, yaitu memilih kelas yang memperoleh suara terbanyak dari seluruh pohon."
        ),
        (
            "4. Hasil Klasifikasi",
            "Berdasarkan hasil penggabungan prediksi, model menentukan kelas akhir "
            "suatu transaksi, misalnya transaksi normal (non-fraud) atau transaksi penipuan " 
            "(fraud), sesuai pola yang dipelajari selama pelatihan."
        )
    ]

    for title, description in steps:
        with st.expander(title):
            st.write(description)

    st.divider()

    # 3. Karakteristik
    st.subheader("Karakteristik Random Forest")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Kelebihan")
        st.success("""
        - Mampu mempelajari hubungan nonlinier.
        - Menggabungkan banyak pohon keputusan.
        - Dapat menyediakan estimasi probabilitas kelas.
        - Mendukung analisis feature importance.
        """)

    with col2:
        st.markdown("#### Keterbatasan")
        st.warning("""
        - Interpretasi model lebih kompleks daripada satu pohon.
        - Membutuhkan sumber daya lebih besar jika pohon banyak.
        - Probabilitas tidak selalu terkalibrasi dengan baik.
        - Tetap dapat salah mengklasifikasikan transaksi.
        """)

    st.divider()

    # 4. Penerapan dalam proyek
    st.subheader("Penerapan pada Fraud Detection")

    st.write("""
        Model menerima fitur transaksi yang telah melalui tahapan preprocessing, termasuk transformasi waktu, scaling fitur
        numerik, dan encoding fitur kategorikal. Selanjutnya, Random Forest memprediksi kelas transaksi (fraud atau non-fraud) 
        berdasarkan pola yang telah dipelajari dari data pelatihan. Hasil prediksi ini dapat digunakan untuk 
        menilai risiko transaksi dan membantu dalam pengambilan keputusan terkait keamanan transaksi.
    """)

    st.info(
        "Hasil klasifikasi merupakan prediksi model berdasarkan "
        "pola data pelatihan, bukan bukti pasti bahwa penipuan terjadi."
    )

# Model Testing
def models():
    st.title("Pengujian Model")

    with st.expander("Input Detail Transaksi"):
        st.markdown("Informasi Transaksi")
        col1, col2 = st.columns(2)

        with col1:
            transaction_amount = st.number_input("💵 Jumlah Transaksi", min_value=0.00, max_value=304.04, value=100.0, step=1.0)
            daily_transaction_count = st.number_input("🔁 Jumlah Transaksi Harian", min_value=1, max_value=14, value=1, step=1)
            avg_transaction_amount_7d = st.number_input("📊 Rata-rata Transaksi 7 Hari", min_value=10.0, max_value=500.0, value=100.0, step=10.0)
            failed_transaction_count_7d = st.number_input("❌ Gagal Transaksi (7 Hari)", min_value=0, max_value=4, value=0, step=1)

        with col2:
            risk_score = st.number_input("⚠ Skor Risiko (0-1)", min_value=0.0001, max_value=0.9999, value=0.1, step=0.01)
            ip_address_flag = st.selectbox("🌐 IP Address Mencurigakan?", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
            location = st.selectbox("📍 Lokasi Transaksi", ['sydney', 'new york', 'mumbai', 'tokyo', 'london'])
            merchant_category = st.selectbox("🏪 Kategori Merchant", ['travel', 'clothing', 'restaurants', 'groceries', 'electronics'])
            authentication_method = st.selectbox("🔐 Metode Autentikasi", ['biometric', 'password', 'otp', 'pin'])

        st.markdown("### 🕒 Waktu Transaksi")
        timestamp_date = st.date_input("📅 Tanggal Transaksi", value=datetime.today().date())
        if 'login_time' not in st.session_state:
            st.session_state.login_time = datetime.now().time()
        login_time = st.time_input("⏰ Waktu Login", value=st.session_state.login_time)
        st.session_state.login_time = login_time
        timestamp = datetime.combine(timestamp_date, login_time)
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    st.divider()

    if st.button("Prediksi Transaksi", use_container_width=True, type="secondary"):
        try:
            input_data = {
                'Transaction_Amount': transaction_amount,
                'IP_Address_Flag': ip_address_flag,
                'Daily_Transaction_Count': daily_transaction_count,
                'Avg_Transaction_Amount_7d': avg_transaction_amount_7d,
                'Failed_Transaction_Count_7d': failed_transaction_count_7d,
                'Risk_Score': risk_score,
                'Location': location,
                'Merchant_Category': merchant_category,
                'Authentication_Method': authentication_method,
                'Timestamp': timestamp_str
            }

            processed = preprocess_input(input_data)
            pred = model.predict(processed)[0]
            proba = model.predict_proba(processed)[0][1]
            display_proba = "100%" if proba >= 0.999 else "0%" if proba <= 0.001 else f"{proba * 100:.2f}%"

            # Output hasil
            st.markdown("## Hasil Prediksi Transaksi")

            # Warna berdasarkan hasil prediksi
            status_color = "#e74c3c" if pred == 1 else "#2ecc71"
            status_icon = "🛑" if pred == 1 else "✅"
            status_title = "Penipuan Terdeteksi!" if pred == 1 else "Transaksi Aman"
            description = (
                "Transaksi ini menunjukkan tanda-tanda mencurigakan. Disarankan untuk <strong>diperiksa ulang</strong> atau <strong>diblokir sementara</strong>."
                if pred == 1
                else "Model tidak menemukan indikasi penipuan. Transaksi dinilai aman untuk diproses."
            )

            # Panel status hasil
            st.markdown(f"""
            <div style='
                background-color: {status_color};
                color: white;
                padding: 1em;
                border-radius: 6px;
                font-family: "Segoe UI", sans-serif;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                margin-top: 1.5em;
            '>
                <div style='font-size: 2em; font-weight: bold; margin-bottom: 0.2em;'>{status_icon} {status_title}</div>
                <div style='font-size: 1.5em; font-weight: bold; margin-bottom: 0.8em;'>
                    Kemungkinan Penipuan: {display_proba}
                </div>
                <div style='font-size: 1.1em; line-height: 1.6;'>{description}</div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Gagal melakukan prediksi:\n\n{e}")

# Model Evaluation
def evaluation():
    st.title("Model Evaluation")

    try:
        _, _, _, _, _, y_test, X_test = load_assets()

        y_pred = model.predict(X_test)

        classes = list(model.classes_)

        if 1 not in classes or 0 not in classes:
            st.error(
                "Model harus memiliki kelas 0 (Non-Fraud) "
                "dan 1 (Fraud) untuk evaluasi ini."
            )
            return

        fraud_index = classes.index(1)
        y_proba = model.predict_proba(X_test)[:, fraud_index]

        y_test = pd.Series(y_test).reset_index(drop=True)

        if not set(y_test.unique()).issubset({0, 1}):
            st.error("Label aktual harus berupa 0 dan 1.")
            return

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, pos_label=1, zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, pos_label=1, zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, pos_label=1, zero_division=0
        )
        roc_auc = roc_auc_score(y_test, y_proba)

        st.divider()
        
        st.subheader("Detail Performa per Kelas (Non-Fraud vs Fraud)")

        report_dict = classification_report(
            y_test, y_pred, 
            target_names=["Non-Fraud (0)", "Fraud (1)"], 
            output_dict=True,
            zero_division=0
        )

        df_report = pd.DataFrame(report_dict).transpose()

        st.dataframe(df_report.style.format("{:.2f}"), use_container_width=True)
        
        st.info(
            "Classifaction report adalah sebuah ringkasan evaluasi model dalam bentuk teks atau tabel yang digunakan "
            "untuk mengukur seberapa baik model mengklasifikasikan data ke dalam kelas tertentu pada model machine learning."
        )

        st.divider()

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_test, y_pred, labels=[0, 1]
        )

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.5))

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Non-Fraud", "Fraud"],
                yticklabels=["Non-Fraud", "Fraud"],
                ax=ax_cm,
                annot_kws={"size": 11}
            )

            ax_cm.set_xlabel("Prediksi Model", fontsize=9)
            ax_cm.set_ylabel("Label Aktual", fontsize=9)
            ax_cm.set_title("Confusion Matrix Random Forest", fontsize=10, pad=10)

            ax_cm.tick_params(axis='x', labelsize=9)
            ax_cm.tick_params(axis='y', labelsize=9)

            plt.tight_layout()
            
            st.pyplot(fig_cm)
            plt.close(fig_cm)
        
        st.info(
            "Confusion Matrix adalah sebuah tabel yang merangkum hasil dari model klasifikasi dengan membandingkan label "
            "prediksi dengan label sebenarnya dari suatu dataset."
        )

        st.markdown(f"""
        - **True Negative (TN):** {cm[0, 0]} transaksi non-fraud
          yang diprediksi non-fraud.
        - **False Positive (FP):** {cm[0, 1]} transaksi non-fraud
          yang diprediksi fraud.
        - **False Negative (FN):** {cm[1, 0]} transaksi fraud
          yang diprediksi non-fraud.
        - **True Positive (TP):** {cm[1, 1]} transaksi fraud
          yang diprediksi fraud.
        """)

        st.divider()

        st.subheader("ROC Curve & ROC-AUC")

        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            
            fpr, tpr, thresholds = roc_curve(
                        y_test, y_proba, pos_label=1
                    )
            
            fig_roc, ax_roc = plt.subplots(figsize=(4.5, 3.5))
    
            ax_roc.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                label="Baseline",
            )
            
            ax_roc.plot(
                fpr,
                tpr,
                label=f"Random Forest (AUC = {roc_auc:.4f})"
            )
            
            ax_roc.set_xlabel("False Positive Rate", fontsize=9)
            ax_roc.set_ylabel("True Positive Rate", fontsize=9)
            ax_roc.set_title("ROC Curve", fontsize=10, pad=10)
            ax_roc.tick_params(axis='both', labelsize=9)
            ax_roc.set_xlim(0, 1)
            ax_roc.set_ylim(0, 1.05)
            ax_roc.legend(loc="lower right", fontsize=7)
            ax_roc.grid(alpha=0.3)

            plt.tight_layout()
            
            st.pyplot(fig_roc)
            plt.close(fig_roc)
            
        st.info(
            "ROC-AUC mengukur kemampuan model membedakan kelas fraud dan non-fraud pada berbagai threshold. "
            "Nilai yang lebih tinggi umumnya menunjukkan kemampuan pemeringkatan yang lebih baik."
        )

    except FileNotFoundError:
        st.warning(
            "File X_test.pkl atau y_test.pkl belum ditemukan. "
            "Simpan data pengujian dari proses pelatihan model "
            "terlebih dahulu."
        )

    except Exception as e:
        st.error(f"Terjadi kesalahan saat evaluasi: {e}")

# Fraud Analysis 
def fraud_analysis():
    st.title("Fraud Analysis")

    try:
        df = pd.read_csv("dataset/synthetic_fraud_dataset.csv")
    except FileNotFoundError:
        st.error("File dataset tidak ditemukan.")
        st.info(
            "Periksa lokasi CSV dan sesuaikan path pada kode."
        )
        return

    target_col = "Fraud_Label"

    if target_col not in df.columns:
        st.error(
            f"Kolom target '{target_col}' tidak ditemukan. "
            f"Kolom tersedia: {', '.join(df.columns)}"
        )
        return

    labels = df[target_col]

    if labels.isin([0, 1]).all():
        fraud_mask = labels.eq(1)
        nonfraud_mask = labels.eq(0)
    else:
        st.error(
            "Contoh kode ini mengharapkan label target 0 dan 1. "
            "Sesuaikan pemetaan label jika dataset menggunakan "
            "teks atau pengodean berbeda."
        )
        return

    total = len(df)
    fraud_count = int(fraud_mask.sum())
    nonfraud_count = int(nonfraud_mask.sum())
    fraud_rate = (
        fraud_count / total * 100 if total else 0
    )

    st.subheader("1. Ringkasan Transaksi")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Transaksi", f"{total:,}")
    col2.metric("Fraud", f"{fraud_count:,}")
    col3.metric("Non-Fraud", f"{nonfraud_count:,}")
    col4.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    
    st.markdown("**Distribusi Label Transaksi**")

    distribution = pd.DataFrame({
    "Kelas Transaksi": ["Non-Fraud", "Fraud"],
    "Jumlah Transaksi": [nonfraud_count, fraud_count]
    })

    st.bar_chart(
        distribution,
        x="Kelas Transaksi",
        y="Jumlah Transaksi",
        color="#3984D6",
        horizontal=False
    )

    st.write("""
        Dataset terdiri dari 50.000 transaksi, dengan 32.13% berlabel fraud dan 67.87% berlabel non-fraud. 
        Pada dataset ini transaksi non-fraud lebih dominan dibandingkan transaksi fraud dengan selisih 17,866 
        transaksi. Meskipun jumlah transaksi non-fraud lebih besar, proporsi transaksi fraud tetap perlu mendapat perhatian 
        karena mencapai hampir sepertiga dari keseluruhan transaksi. Kondisi ini menunjukkan bahwa transaksi penipuan merupakan 
        bagian yang cukup besar dalam dataset, sehingga penting untuk mengidentifikasi pola transaksi fraud dan 
        memahami karakteristiknya guna mendukung upaya pengelolaan risiko penipuan.
    """)

    st.divider()

    # Merchant category
    st.subheader("2. Analisis Fraud berdasarkan Merchant")

    if "Merchant_Category" in df.columns:
        merchant_rate = (
            df.assign(_fraud=fraud_mask.astype(int))
            .groupby("Merchant_Category")["_fraud"]
            .agg(["sum", "count", "mean"])
            .reset_index()
        )

        merchant_rate["Fraud Rate (%)"] = merchant_rate["mean"] * 100

        merchant_rate = merchant_rate.sort_values(
            "Fraud Rate (%)",
            ascending=False
        )

        fig, ax = plt.subplots(figsize=(8, 4))

        for _, row in merchant_rate.sort_values(
            "Fraud Rate (%)", ascending=False
        ).iterrows():
            rate = row["Fraud Rate (%)"]

            st.write(f'{row["Merchant_Category"]} — {rate:.2f}%')
            st.progress(min(max(float(rate) / 100, 0), 1))

        st.markdown("**Ringkasan Fraud Rate per Kategori Merchant**")
        
        st.dataframe(
            merchant_rate[
                ["Merchant_Category", "sum", "count", "Fraud Rate (%)"]
            ].rename(columns={
                "Merchant_Category": "Kategori Merchant",
                "sum": "Jumlah Fraud",
                "count": "Total Transaksi"
            }).assign(
                **{"Fraud Rate (%)": lambda x: x["Fraud Rate (%)"].round(2)}
            ),
            use_container_width=True,
            hide_index=True
        )

        if not merchant_rate.empty:
            top = merchant_rate.iloc[0]

            st.info(
                f"Kategori dengan fraud rate tertinggi adalah "
                f"{top['Merchant_Category']} sebesar "
                f"{top['Fraud Rate (%)']:.2f}%, dari "
                f"{int(top['count']):,} transaksi."
            )

    st.write("""
        Penyebab dari penipuan transaksi dapat bervariasi berdasarkan kategori merchant. Analisis ini membantu mengidentifikasi merchant 
        yang lebih rentan terhadap penipuan, sehingga dapat menjadi fokus pengawasan dan mitigasi risiko. Berasarkan 
        data, kategori merchant **Restaurants** menunjukkan fraud rate tertinggi di antara kategori dalam dataset, tetapi 
        selisihnya relatif kecil dibandingkan kategori lainnya.
    """)
    st.divider()

    #Transaction amount
    st.subheader("3. Analisis Nominal Transaksi")

    amount_col = "Transaction_Amount"

    if amount_col in df.columns:
        data_amount = df[[amount_col]].copy()

        data_amount["Label"] = labels.map({
            0: "Non-Fraud",
            1: "Fraud"
        })

        # Pastikan nominal berupa numerik
        data_amount[amount_col] = pd.to_numeric(
            data_amount[amount_col],
            errors="coerce"
        )

        data_amount = data_amount.dropna(
            subset=[amount_col, "Label"]
        )
    else:
        st.info("Kolom Transaction_Amount tidak tersedia.")
        st.stop()

    # Interval nominal yang SAMA untuk kedua label
    batas_nominal = [
        0, 25, 50, 75, 100, 150, 200, 300,
        500, 750, 1000, float("inf")
    ]

    label_interval = [
        "0–25", "25–50", "50–75", "75–100",
        "100–150", "150–200", "200–300",
        "300–500", "500–750", "750–1000",
        ">1000"
    ]

    data_interval = data_amount.copy()

    data_interval["Interval Nominal"] = pd.cut(
        data_interval[amount_col],
        bins=batas_nominal,
        labels=label_interval,
        right=False,
        include_lowest=True
    )

    tabel_jumlah = pd.crosstab(
        data_interval["Interval Nominal"],
        data_interval["Label"]
    ).reindex(label_interval, fill_value=0)

    for label in ["Fraud", "Non-Fraud"]:
        if label not in tabel_jumlah.columns:
            tabel_jumlah[label] = 0

    tabel_jumlah = tabel_jumlah[["Non-Fraud", "Fraud"]]

    tabel_jumlah["Total Transaksi"] = (
        tabel_jumlah["Non-Fraud"] + tabel_jumlah["Fraud"]
    )

    tabel_jumlah["Proporsi Fraud (%)"] = (
        tabel_jumlah["Fraud"]
        .div(tabel_jumlah["Total Transaksi"].replace(0, float("nan")))
        .mul(100)
    )
    
    plot_data = (
        tabel_jumlah["Proporsi Fraud (%)"]
        .dropna()
        .reset_index()
    )
    
    plot_data = plot_data.sort_index()

    fig, ax = plt.subplots(figsize=(10, 4))

    st.bar_chart(
        plot_data.set_index("Interval Nominal")[["Proporsi Fraud (%)"]],
        x_label="Interval Nominal",
        y_label="Proporsi Fraud (%)",
        sort=False
    )

    st.write("""
        Visualisasi di atas menunjukkan bahwa pada rentang nominal transaksi 0–500 dolar proporsi transaksi penipuan 
        berada di angka 30%, tetapi saat nominal transaksi meningkat hingga 750–1.000 dolar proporsi 
        penipuan juga ikut meningkat hingga mencapai 50%. Hal ini mengindikasikan bahwa transaksi dengan nominal 
        yang lebih besar memiliki kemungkinan penipuan yang lebih tinggi, tetapi tidak dapat langsung disimpulkan 
        bahwa transaksi bernominal besar pasti merupakan penipuan. Visualisasi ini hanya menunjukkan proporsi transaksi penipuan 
        pada setiap rentang nominal, bukan jumlah keseluruhan transaksi dalam rentang tersebut.
    """)

    st.markdown("**Ringkasan Fraud Rate per Nominal Transaksi**")
    
    st.dataframe(
        tabel_jumlah.round(2),
        use_container_width=True
    )
    
    st.write("""
        Hasil visualisasi sebelumnya menunjukkan bahwa rentang nominal transaksi 750–1.000 dolar memiliki proporsi penipuan yang 
        relatif tinggi, yaitu sekitar 40–50%. Berdasarkan ringkasan pada tabel, tingginya proporsi tersebut disebebabkan 
        oleh jumlah transaksi yang relatif sedikit menyebabkan proporsi penipuan terlihat tinggi, sehingga hasil tersebut 
        perlu diinterpretasikan secara hati-hati.

        \nBerbeda dengan rentang nominal 0–500 dolar  proporsi penipuan pada rentang ini relatif konsisten, yaitu 
        sekitar 30%, dengan jumlah transaksi yang mencapai ribuan. Banyaknya transaksi tersebut memberikan gambaran yang 
        lebih luas mengenai pola penipuan pada rentang nominal tersebut. Berdasarkan ringkasan pada tabel, transaksi penipuan 
        paling banyak ditemukan pada rentang nominal 0–25 dolar, dengan jumlah mencapai 3.603 transaksi, lalu 
        rentang 25–50 dolar menempati urutan kedua dengan 2.731 transaksi penipuan. Temuan ini menunjukkan bahwa 
        transaksi bernominal kecil juga perlu mendapat perhatian dalam analisis risiko penipuan, mengingat jumlah kasus 
        yang ditemukan cukup besar. 
    """)

    st.divider()

    # Time analysis
    st.subheader("4. Analisis Waktu Transaksi")

    if "Timestamp" in df.columns:
        time_df = df.copy()
        time_df["Timestamp"] = pd.to_datetime(
            time_df["Timestamp"],
            errors="coerce"
        )
        time_df = time_df.dropna(subset=["Timestamp"])
        time_df["Jam"] = time_df["Timestamp"].dt.hour
        time_df["_fraud"] = fraud_mask.loc[
            time_df.index
        ].astype(int)

        hourly = (
            time_df.groupby("Jam")["_fraud"]
            .agg(["sum", "count"])
            .reindex(range(24), fill_value=0)
        )

        hourly["Fraud Rate (%)"] = (
            hourly["sum"].div(
                hourly["count"].replace(0, float("nan"))
            ) * 100
        )

        fig, ax = plt.subplots(figsize=(9, 4))
        hourly["Fraud Rate (%)"].plot(
            kind="line",
            marker="o",
            ax=ax
        )
        
        st.altair_chart(
            alt.Chart(hourly.reset_index()).mark_line(point=True).encode(
                x=alt.X("Jam:Q", title="Jam Transaksi", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("Fraud Rate (%):Q", title="Fraud Rate (%)", scale=alt.Scale(domain=[25, 35])),
                tooltip=["Jam", alt.Tooltip("Fraud Rate (%)", format=".2f")]
            ).properties(height=350),
            use_container_width=True
        )
        
        st.write(""" 
            Visualisasi diatas menunjukan bahwa proporsi transaksi penipuan pada sebagian besar waktu berada di kisaran 
            30%-33% tetapi pada pukul 21.00 penipuan turun hingga dibawah 28,7% sedangkan nilai tertinggi terjadi 
            pada pukul 22.00, yaitu sekitar 33,8%. Meskipun perbedaan persentase tersebut terlihat relatif kecil, perubahan 
            ini tetap menarik untuk diperhatikan karena dapat berkaitan dengan jumlah transaksi yang terlibat. Persentase 
            saja belum cukup untuk menentukan besarnya dampak penurunan tersebut, sehingga jumlah transaksi pada setiap 
            jam juga perlu dipertimbangkan. Grafik juga menunjukkan adanya fluktuasi fraud rate sepanjang hari, tetapi 
            belum memperlihatkan pola peningkatan atau penurunan yang konsisten berdasarkan waktu.  
        """)

        # Tabel Fraud Rate per Jam
        
        st.markdown("**Ringkasan Fraud Rate per Jam Transaksi**")
        
        fraud_table = hourly.reset_index().rename(columns={
            "Jam": "Jam Transaksi",
            "sum": "Jumlah Fraud",
            "count": "Total Transaksi"
        })

        fraud_table = fraud_table[
            fraud_table["Total Transaksi"] > 0
        ].copy()
        
        fraud_filtered = fraud_table[
            (fraud_table["Fraud Rate (%)"] >= 28) &
            (fraud_table["Fraud Rate (%)"] <= 34)
        ].copy()

        fraud_filtered["Jam Transaksi"] = (
            fraud_filtered["Jam Transaksi"].astype(str)
            .str.zfill(2) + ":00"
        )

        fraud_filtered["Fraud Rate (%)"] = (
            fraud_filtered["Fraud Rate (%)"].round(2)
        )

        fraud_filtered = fraud_filtered.sort_values(
            "Fraud Rate (%)"
        )

        if not fraud_filtered.empty:
            st.dataframe(
                fraud_filtered[
                    [
                        "Jam Transaksi",
                        "Jumlah Fraud",
                        "Total Transaksi",
                        "Fraud Rate (%)",
                    ]
                ],
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info(
                "Tidak ada jam transaksi dengan fraud rate "
                "antara 28% hingga 34%."
            )
        st.write(""" 
            Berdasarkan hasil visualisasi sebelumnya, proporsi penipuan tertinggi terjadi pada pukul 22.00, yaitu sekitar 33,8%. 
            Jika ditinjau lebih lanjut melalui ringkasan pada tabel, pukul 22.00 tidak hanya menunjukkan proporsi penipuan 
            yang tinggi, tetapi juga memiliki jumlah transaksi penipuan tertinggi kedua. Periode pukul 20.00 hingga 
            22.00 juga menunjukkan proporsi penipuan yang relatif tinggi, yaitu  berkisar antara 33,13% hingga 33,76% tetapi menjadi 
            rendah pada pukul 21.00 yang berada di angka 28,72%. Walaupun begitu tetap perlu perhatian 
            mengingat tingginya proporsi penipuan transaksi yang relatif besar. Hal tersebut membuat analisis tidak cukup 
            hanya berfokus pada proporsi penipuan, tetapi juga perlu mempertimbangkan jumlah transaksi penipuan yang terjadi 
            pada setiap jam. 
            
            \nBerdasarkan ringkasan pada tabel, jumlah transaksi penipuan yang tinggi tidak hanya ditemukan pada malam hari, 
            tetapi juga pada pukul 10.00 hingga 13.00 dengan rata-rata transaksi penipuan yang terjadi berjumlah 
            692. Temuan ini menunjukkan bahwa periode dengan jumlah transaksi penipuan yang perlu diperhatikan mencakup pukul 10.00–13.00 
            serta pukul 20.00–22.00. Kedua periode tersebut dapat menjadi fokus analisis lebih lanjut karena menunjukkan 
            proporsi penipuan yang relatif tinggi, dengan mempertimbangkan jumlah transaksiyang terjadi pada masing-masing periode.
        """)
    else:
        st.info("Kolom Timestamp tidak tersedia.")
        
    st.divider()
        
    st.subheader("5. Kesimpulan Analisis Fraud")

    st.write("""
        Kategori merchant Restaurants memiliki fraud rate tertinggi dibandingkan kategori lainnya, meskipun selisihnya relatif kecil 
        kategori ini tetap dipilih untuk analisis lebih lanjut guna mengidentifikasi pola penipuan. Pemilihan tersebut didasarkan pada 
        hasil analisis deskriptif dan merupakan prioritas untuk pendalaman analisis.
    """)

    df_restaurant = df[
        df["Merchant_Category"].astype(str).str.strip().str.lower() == "restaurants"
    ].copy()

    if df_restaurant.empty:
        st.warning("Tidak ditemukan data untuk kategori merchant Restaurants.")
    else:   
        df_restaurant["Fraud_Label"] = pd.to_numeric(
            df_restaurant["Fraud_Label"], errors="coerce"
        )
        df_restaurant["Transaction_Amount"] = pd.to_numeric(
            df_restaurant["Transaction_Amount"], errors="coerce"
        )

        df_restaurant = df_restaurant.dropna(
            subset=["Fraud_Label", "Transaction_Amount"]
        )

        total_transaksi = len(df_restaurant)
        total_fraud = (df_restaurant["Fraud_Label"] == 1).sum()
        total_non_fraud = (df_restaurant["Fraud_Label"] == 0).sum()
        fraud_rate = (
            total_fraud / total_transaksi * 100
            if total_transaksi > 0 else 0
        )

        st.markdown("### Ringkasan Merchant Restaurants")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transaksi", f"{total_transaksi:,}")
        col2.metric("Transaksi Fraud", f"{total_fraud:,}")
        col3.metric("Transaksi Non-Fraud", f"{total_non_fraud:,}")
        col4.metric("Fraud Rate", f"{fraud_rate:.2f}%")

        st.write(
            f"Dari **{total_transaksi:,} transaksi** pada kategori Restaurants, "
            f"terdapat **{total_fraud:,} transaksi fraud** dengan fraud rate "
            f"sebesar **{fraud_rate:.2f}%**."
        )

        df_restaurant_fraud = df_restaurant[
            df_restaurant["Fraud_Label"] == 1
        ].copy()

        st.markdown("### Analisis Nominal Transaksi Restaurants")

        fig_amount, ax_amount = plt.subplots(figsize=(9, 5))

        bins = [0, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, float("inf")]
        labels = [
            "0–25", "25–50", "50–75", "75–100", "100–150",
            "150–200", "200–300", "300–500", "500–750",
            "750–1000", ">1000"
        ]

        df_restaurant["Interval Nominal"] = pd.cut(
            df_restaurant["Transaction_Amount"],
            bins=bins,
            labels=labels,
            include_lowest=True
        )

        plot_data = (
            df_restaurant.groupby("Interval Nominal", observed=False)
            .agg(
                Total_Transaksi=("Fraud_Label", "count"),
                Jumlah_Fraud=("Fraud_Label", lambda x: (x == 1).sum())
            )
            .reset_index()
        )

        plot_data["Proporsi Fraud (%)"] = (
            plot_data["Jumlah_Fraud"] / plot_data["Total_Transaksi"] * 100
        )

        st.bar_chart(
            plot_data.set_index("Interval Nominal")[["Proporsi Fraud (%)"]],
            x_label="Interval Nominal",
            y_label="Proporsi Fraud (%)",
            sort=False
        )
        
        st.write("""
            Visualisasi diatas menunjukan bahwa nominal transaksi yang dilakukan pada rentang 0-500 dolar memiliki pola 
            yang konsisten di angka 30%, namun saat nominal transaksi meningkat hingga 500-1000 dolar angka penipuan cukup 
            menurun menjadi sekitar 25%, atau berkurang sekitar 5 poin persentase dibandingkan rentang sebelumnya. Meskipun 
            secara selisih absolut angkanya tampak kecil, secara persentase penurunan risiko/proporsi penipuan sebesar 
            ~16,7% saat nominal transaksi berada pada rentang 500–1.000 dolar dibandingkan dengan rentang 0–500 dolar 
            umumnya dianggap perubahan yang berarti dalam analisis perilaku transaksi.
        """)

        st.markdown("### Analisis Waktu Transaksi Restaurants")

        df_restaurant["Timestamp"] = pd.to_datetime(
            df_restaurant["Timestamp"], errors="coerce"
        )
        df_restaurant = df_restaurant.dropna(subset=["Timestamp"])
        df_restaurant["Hour"] = df_restaurant["Timestamp"].dt.hour

        fraud_per_hour = (
            df_restaurant[df_restaurant["Fraud_Label"] == 1]
            .groupby("Hour")
            .size()
            .reindex(range(24), fill_value=0)
        )

        total_per_hour = (
            df_restaurant.groupby("Hour")
            .size()
            .reindex(range(24), fill_value=0)
        )

        fraud_rate_per_hour = (
            fraud_per_hour.div(total_per_hour.mask(total_per_hour == 0)) * 100
        )

        hourly_df = pd.DataFrame({
            "Jam": range(24),
            "Jumlah Transaksi": total_per_hour.values,
            "Jumlah Fraud": fraud_per_hour.values,
            "Fraud Rate (%)": fraud_rate_per_hour.values
        })

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Jumlah Transaksi Fraud per Jam**")

            chart_count = alt.Chart(hourly_df).mark_line(point=True).encode(
                x=alt.X(
                    "Jam:Q",
                    title="Jam Transaksi",
                    axis=alt.Axis(tickMinStep=2)
                ),
                y=alt.Y(
                    "Jumlah Fraud:Q",
                    title="Jumlah Fraud",
                    scale=alt.Scale(domain=[105, 170])
                ),
                tooltip=[
                    alt.Tooltip("Jam:Q", title="Jam", format=".0f"),
                    alt.Tooltip("Jumlah Fraud:Q", title="Jumlah Fraud", format=",")
                ]
            ).properties(
                height=350
            )

            st.altair_chart(
                chart_count,
                use_container_width=True
            )

        with col2:
            st.markdown("**Fraud Rate per Jam**")

            chart_rate = alt.Chart(hourly_df).mark_line(point=True).encode(
                x=alt.X(
                    "Jam:Q",
                    title="Jam Transaksi",
                    axis=alt.Axis(tickMinStep=2)
                ),
                y=alt.Y(
                    "Fraud Rate (%):Q",
                    title="Fraud Rate (%)",
                    scale=alt.Scale(domain=[28, 38])
                ),
                tooltip=[
                    alt.Tooltip("Jam:Q", title="Jam", format=".0f"),
                    alt.Tooltip("Fraud Rate (%):Q", title="Fraud Rate", format=".2f")
                ]
            ).properties(
                height=350
            )

            st.altair_chart(
                chart_rate,
                use_container_width=True
            )
            
        st.write("""
            Terdapat beberapa pola yang dapat diamati dari kedua visualisasi di atas. Berdasarkan jumlah transaksi penipuan, 
            pada pukul 00.00 jumlah transaksi fraud berada di kisaran 125 transaksi, kemudian menurun hingga sekitar 110
            transaksi pada pukul 03.00. Selanjutnya, jumlah transaksi penipuan meningkat hingga sekitar 150 transaksi pada
            pukul 06.00 dan relatif stabil pada pukul 07.00 hingga pukul 14.00. Pada pukul 14.00, jumlah transaksi
            penipuan mengalami penurunan, kemudian meningkat kembali pada pukul 15.00 hingga 19.00. Setelah pukul 19.00,
            jumlah transaksi penipuan kembali mengalami penurunan yang cukup signifikan. Pola ini menunjukkan bahwa jumlah
            transaksi penipuan meningkat pada pukul 06.00 serta pukul 15.00–19.00 dan cenderung stabil pada pukul 07.00-14.00, 
            sehingga kedua rentang waktu tersebut perlu diperhatikan dalam analisis lebih lanjut.
            
            \nJika ditinjau berdasarkan proporsi penipuan, pola yang terlihat menunjukkan beberapa perbedaan. Pada pukul
            00.00 hingga 03.00, proporsi penipuan menurun dari sekitar 31% menjadi 28%. Selanjutnya, proporsi tersebut 
            meningkat pada pukul 04.00 hingga 09.00 dan mencapai puncaknya pada pukul 09.00, yaitu sekitar 36%. Setelah 
            itu, proporsi penipuan menurun pada pukul 10.00 hingga 12.00. Proporsi penipuan kembali meningkat pada pukul 
            13.00 hingga 19.00, sebelum mengalami penurunan menjelang pukul 20.00. Pola ini menunjukkan bahwa pukul 09.00
            memiliki proporsi penipuan tertinggi, sedangkan peningkatan pada pukul 13.00–19.00 juga perlu diperhatikan karena
            terjadi selama beberapa jam berturut-turut. 
            
            \nBerdasarkan kedua visualisasi tersebut, pukul 03.00–09.00 dan pukul 15.00–19.00 dapat menjadi fokus analisis 
            lebih lanjut. Pada rentang pukul 03.00–09.00, proporsi penipuan berada di kisaran 32,1%, sedangkan pada rentang 
            pukul 15.00–19.00 proporsinya mencapai sekitar 33,96%.  Hal ini dilakukan guna memudahkan analisis agar lebih fokus 
            pada rentang waktu yang memiliki angka penipuan tinggi.
        """)
        
        st.markdown("### Rincian Fraud dan Nominal Transaksi per Jam")

        jam_fokus = list(range(3, 10)) + list(range(15, 20))

        df_fokus = df_restaurant[
            df_restaurant["Hour"].isin(jam_fokus)
        ].copy()

        ringkasan_jam = (
            df_fokus.groupby("Hour")
            .agg(
                Total_Transaksi=("Fraud_Label", "count"),
                Jumlah_Fraud=("Fraud_Label", lambda x: (x == 1).sum()),
                Total_Nominal_Fraud=(
                    "Transaction_Amount",
                    lambda x: x[df_fokus.loc[x.index, "Fraud_Label"] == 1].sum()
                )
            )
            .reindex(jam_fokus, fill_value=0)
            .reset_index()
        )

        ringkasan_jam["Fraud Rate (%)"] = (
            ringkasan_jam["Jumlah_Fraud"]
            / ringkasan_jam["Total_Transaksi"].replace(0, np.nan)
            * 100
        )
        
        ringkasan_jam["Jam"] = ringkasan_jam["Hour"].map(
            lambda h: f"{h:02d}:00"
        )

        ringkasan_jam = ringkasan_jam.sort_values(
            "Total_Nominal_Fraud", ascending=False
        )
        
        plot_nominal = (
            ringkasan_jam
            .sort_values("Hour")
            .set_index("Jam")[["Total_Nominal_Fraud"]]
            .rename(columns={
                "Total_Nominal_Fraud": "Total Nominal Fraud"
            })
        )
    
        st.bar_chart(
            plot_nominal,
            x_label="Jam Transaksi",
            y_label="Total Nominal Transaksi Fraud",
            sort=False
        )
        
        st.write("""
            Grafik di atas menunjukkan rata-rata nominal transaksi yang terindikasi fraud pada setiap jam. 
            Rata-rata nominal transaksi fraud berada di kisaran 10.000 Dolar, meskipun pada pukul 03.00 rata-rata 
            nominalnya lebih rendah, yaitu sekitar 8.000 Dolar, transaksi penipuan tetap perlu mendapat perhatian karena 
            berpotensi menimbulkan kerugian. Pada pukul 15.00 hingga 19.00 rata-rata nominal transaksi fraud mencapai 
            lebih dari 14.000 Dolar, yang mana lebih tinggi jika dibandingkan pukul 03.00. Guna memahami distribusi nominal transaksi 
            fraud secara lebih menyeluruh, diperlukan analisis statistik deskriptif terhadap data tersebut.
        """)

        st.markdown("**Ringkasan Nominal Transaksi Fraud per Jam**")
        
        st.dataframe(
            ringkasan_jam[
                [
                    "Jam",
                    "Total_Transaksi",
                    "Jumlah_Fraud",
                    "Fraud Rate (%)",
                    "Total_Nominal_Fraud"
                ]
            ].rename(columns={
                "Total_Transaksi": "Total Transaksi",
                "Jumlah_Fraud": "Jumlah Fraud",
                "Fraud Rate (%)": "Fraud Rate (%)",
                "Total_Nominal_Fraud": "Total Nominal Fraud"
            }).round(2),
            use_container_width=True,
            hide_index=True
        )
        
        st.write("""
            Berdasarkan ringkasan tabel nominal transaksi fraud per Jam pada kategori merchant **Restaurants**, rentang pukul 
            03.00–09.00 yang mencakup tujuh jam memiliki total nilai transaksi penipuan yang cukup besar dan rentang pukul 15.00–19.00 
            hanya mencakup lima jam. Meskipun total nilai transaksi penipuan pada rentang tujuh jam mencapai 92.805,36 Dolar, 
            rata-rata nilai transaksi penipuan per jam pada rentang lima jam justru lebih tinggi, yaitu sekitar 15.000 dolar per jam, 
            dibandingkan dengan 13.000 dolar per jam pada rentang tujuh jam. Temuan ini menunjukkan bahwa meskipun rentang 
            pukul 03.00–09.00 memiliki durasi pengamatan yang lebih panjang, nilai transaksi penipuan rata-rata per jam 
            pada pukul 15.00–19.00 lebih besar.
        """)
        st.markdown("### Kesimpulan dan Rekomendasi")

        st.write("""
            Berdasarkan hasil analisis melalui visualisasi grafik dan statistik deskriptif, kategori merchant **Restaurants** 
            menunjukkan proporsi penipuan sebesar 33,89% pada rentang pukul 15.00–19.00. Pada periode tersebut, rata-rata 
            nilai transaksi penipuan mencapai sekitar 15.000 dolar per jam selama lima jam pengamatan dari total keseluruhan transaksi. 
            Temuan ini menunjukkan bahwa periode tersebut perlu mendapat perhatian karena memiliki proporsi penipuan yang relatif 
            tinggi dan rata-rata nilai transaksi penipuan yang cukup besar.

            **Rekomendasi:** Pengawasan transaksi pada kategori merchant Restaurants dapat diprioritaskan pada pukul 15.00–19.00, 
            terutama untuk mengidentifikasi pola transaksi yang berpotensi terkait dengan penipuan. Evaluasi berkala 
            terhadap jumlah transaksi fraud dan nilai kerugian pada periode tersebut diperlukan untuk menentukan prioritas 
            mitigasi risiko. Karena analisis ini menggunakan data sintetis, temuan tersebut masih bersifat indikatif 
            dan perlu divalidasi menggunakan data transaksi aktual sebelum diterapkan sebagai dasar pengambilan keputusan bisnis.
        """)
    
# Sidebar Navigation
def main():
    st.sidebar.title("Fraud Detection Dashboard")
    st.sidebar.caption("Machine Learning - Random Forest")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Pilih Halaman",
        [
            "Dataset Overview",
            "Algorithm",
            "Model",
            "Evaluation",
            "Fraud Analysis"
        ],
        index=0
    )

    # Navigasi halaman
    if page == "Dataset Overview":
        dataset()
    
    elif page == "Algorithm":
        algorithm()
            
    elif page == "Model":
        models()
            
    elif page == "Evaluation":
        evaluation()                 

    elif page == "Fraud Analysis":
        fraud_analysis()  

main()