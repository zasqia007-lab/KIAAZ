import streamlit as st
import joblib
import numpy as np

# Set judul dan konfigurasi halaman
st.set_page_config(page_title="Melbourne House Price Predictor", layout="centered")

st.title("🏡 Melbourne House Price Predictor")
st.write("Masukkan spesifikasi properti di bawah ini untuk memprediksi estimasi harga rumah.")

# Load model joblib yang sudah diunggah ke Colab
@st.cache_resource
def load_model():
    # Pastikan file 'melbourne_model.joblib' sudah di-upload ke direktori Colab Anda
    return joblib.load('melbourne_model.joblib')

try:
    model = load_model()

    # Membuat form input untuk fitur-fitur yang diminta
    st.header("Spesifikasi Rumah")

    col1, col2 = st.columns(2)
    with col1:
        rooms = st.number_input("Rooms (Jumlah Kamar)", min_value=1, max_value=10, value=3, step=1)
        bathroom = st.number_input("Bathroom (Jumlah Kamar Mandi)", min_value=1, max_value=10, value=2, step=1)
        landsize = st.number_input("Landsize (Luas Tanah dalam m²)", min_value=0, value=500, step=10)

    with col2:
        latitude = st.number_input("Latitude", format="%.5f", value=-37.8136)
        longitude = st.number_input("Longitude", format="%.5f", value=144.9631)

    # Tombol Prediksi
    if st.button("Prediksi Harga Rumah", type="primary"):
        # Menyusun data input sesuai dengan urutan fitur pada model Anda
        # Urutan: Rooms, Bathroom, Landsize, Lattitude, Longtitude
        input_data = np.array([[rooms, bathroom, landsize, latitude, longitude]])

        # Melakukan prediksi
        prediction = model.predict(input_data)

        # Menampilkan hasil prediksi
        st.success("### Hasil Prediksi")
        # Format ke mata uang (Contoh: AUD atau Rupiah, sesuaikan dengan target model Anda)
        st.metric(label="Estimasi Harga Properti", value=f"${prediction[0]:,.2f}")

except FileNotFoundError:
    st.error("⚠️ File 'melbourne_model.joblib' tidak ditemukan. Silakan upload file model Anda ke panel files di sebelah kiri Google Colab terlebih dahulu.")
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model atau memprediksi: {e}")
