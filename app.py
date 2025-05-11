import streamlit as st
import joblib
import os
import cv2
import numpy as np
from PIL import Image

# Module riêng
from shape_recognition import run_shape_recognition
from face_recog.face_recog import run_real_time_recognition
from face_recog.face_recog import run_face_recog
from face_recog.Chuong349 import st_chapter3 as c3
from face_recog.Chuong349 import st_chapter4 as c4
from face_recog.Chuong349 import st_chapter9 as c9

# Cấu hình trang
st.set_page_config(page_title="Dự án Thị giác máy", layout="wide")
st.title("22146295-HoangMinhDuc - 22146291-LePhatDat")

# Đường dẫn mô hình và scaler
model_path = "D:/TESTCODE TGM/PROJECT/pages/face_recog/face_recognition_model.pkl"
scaler_path = "D:/TESTCODE TGM/PROJECT/pages/face_recog/scaler.pkl"

# Tải model
if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    st.success("✅ Mô hình và scaler đã được tải thành công.")
else:
    model = None
    scaler = None
    st.error("❌ Không tìm thấy mô hình hoặc scaler. Kiểm tra lại đường dẫn.")

# Menu chính
menu = st.sidebar.selectbox("📂 Chọn chức năng", [
    "Trang chủ",
    "Nhận dạng khuôn mặt",
    "Nhận dạng hình học (Shape)",
    "Nhận dạng trái cây (YOLOv11n)",
    "Lý thuyết thị giác máy (Chương 3,4,9)",
    "Mở rộng thêm (tuỳ chọn)"
])

if menu == "Trang chủ":
    st.subheader("📌 Giới thiệu")
    st.write("Chào mừng bạn đến với dự án Thị giác máy - sử dụng Streamlit.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/56/Computer_vision.svg", width=400)

elif menu == "Nhận dạng khuôn mặt":
    st.subheader("🧑‍🤝‍🧑 Nhận dạng khuôn mặt")

    # Lựa chọn giữa việc nhận diện khuôn mặt từ ảnh tải lên hoặc từ camera
    choice = st.radio("Chọn phương thức nhận dạng khuôn mặt:", ["Nhận diện từ ảnh tải lên", "Nhận diện từ camera"])

    # Nhận diện từ ảnh hoặc video tải lên
    if choice == "Nhận diện từ ảnh tải lên":
        uploaded_file = st.file_uploader("📤 Tải ảnh hoặc video để nhận diện khuôn mặt", type=["jpg", "jpeg", "png", "tif", "mp4"])
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

            if uploaded_file.type.startswith("image"):
                # Xử lý ảnh
                image = Image.open(uploaded_file)
                image = np.array(image)  # Chuyển ảnh thành mảng NumPy
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Chuyển đổi ảnh từ RGB sang BGR nếu cần thiết

                if model and scaler:
                    run_face_recog(model=model, scaler=scaler, image=image_bgr)  # Gọi hàm nhận dạng khuôn mặt
                else:
                    st.error("❌ Thiếu mô hình hoặc scaler.")
            
            elif uploaded_file.type == "video/mp4":
                # Xử lý video
                if model and scaler:
                    run_face_recog(model=model, scaler=scaler, video_bytes=file_bytes)  # Gọi hàm nhận dạng khuôn mặt cho video
                else:
                    st.error("❌ Thiếu mô hình hoặc scaler.")

    # Nhận diện từ camera
    elif choice == "Nhận diện từ camera":
        if model and scaler:
            run_real_time_recognition(model=model, scaler=scaler)  # Gọi hàm nhận dạng khuôn mặt từ camera
        else:
            st.error("❌ Thiếu mô hình hoặc scaler.")
elif menu == "Nhận dạng hình học (Shape)":
    st.subheader("🔷 Nhận dạng hình học")
    run_shape_recognition()

elif menu == "Nhận dạng trái cây (YOLOv11n)":
    st.subheader("🍎 Nhận dạng trái cây bằng YOLOv11n")
    st.write("🚧 Tính năng đang phát triển...")

elif menu == "Lý thuyết thị giác máy (Chương 3,4,9)":
    st.subheader("📘 Lý thuyết thị giác máy")
    chapter = st.selectbox("📚 Chọn chương", [
        "Chương 3: Xử lý ảnh cơ bản",
        "Chương 4: Phát hiện tần số & lọc nhiễu",
        "Chương 9: Nhận dạng đối tượng"
    ])

    # Chương 3
    if chapter == "Chương 3: Xử lý ảnh cơ bản":
        st.markdown("### 📍 Chương 3: Xử lý ảnh cơ bản")
        uploaded_file = st.file_uploader("📤 Tải ảnh (Chương 3)", type=["jpg","jpeg","png","tif"], key="c3")
        if uploaded_file:
            image = Image.open(uploaded_file)
            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            option3 = st.selectbox("🔧 Phương pháp:", [
                "Negative (Gray)", "Negative (Color)", "Logarit",
                "Power (Gamma)", "Piecewise Linear", "Histogram",
                "Histogram Equalization", "Local Histogram",
                "Histogram Statistic", "Sharpening", "Gradient"
            ], key="opt3")

            # Xử lý ảnh ngay
            if option3 == "Negative (Gray)":
                result = c3.Negative(img_gray)
            elif option3 == "Negative (Color)":
                result = c3.NegativeColor(img_bgr)
            elif option3 == "Logarit":
                result = c3.Logarit(img_gray)
            elif option3 == "Power (Gamma)":
                result = c3.Power(img_gray)
            elif option3 == "Piecewise Linear":
                result = c3.PiecewiseLine(img_gray)
            elif option3 == "Histogram":
                result = c3.Histogram(img_gray)
            elif option3 == "Histogram Equalization":
                result = c3.HistEqual(img_gray)
            elif option3 == "Local Histogram":
                result = c3.LocalHist(img_gray)
            elif option3 == "Histogram Statistic":
                result = c3.HistStat(img_gray)
            elif option3 == "Sharpening":
                result = c3.Sharp(img_gray)
            elif option3 == "Gradient":
                result = c3.Gradien(img_gray)
            else:
                result = img_gray

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Ảnh gốc", use_container_width=True)
            with col2:
                st.image(result, caption="Kết quả", use_container_width=True)

    # Chương 4
    elif chapter == "Chương 4: Phát hiện tần số & lọc nhiễu":
        st.markdown("### 📍 Chương 4: Phát hiện tần số & lọc nhiễu")
        uploaded_file_4 = st.file_uploader("📤 Tải ảnh (Chương 4)", type=["jpg","jpeg","png","tif"], key="c4")
        if uploaded_file_4:
            image4 = Image.open(uploaded_file_4).convert("L")
            img4 = np.array(image4)

            st.image(image4, caption="Ảnh gốc (xám)", use_container_width=True)

            option4 = st.selectbox("🔧 Phương pháp Chương 4:", [
                "Spectrum",
                "DrawNotchFilter",
                "DrawNotchPeriodFilter",
                "RemoveMoireSimple",
                "RemovePeriodNoise"
            ], key="opt4")

            if option4 == "Spectrum":
                result4 = c4.Spectrum(img4)
            elif option4 == "DrawNotchFilter":
                result4 = c4.DrawNotchFilter(img4)
            elif option4 == "DrawNotchPeriodFilter":
                result4 = c4.DrawNotchPeriodFilter(img4)
            elif option4 == "RemoveMoireSimple":
                result4 = c4.RemoveMoireSimple(img4)
            else:  # RemovePeriodNoise
                result4 = c4.RemovePeriodNoise(img4)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image4, caption="Ảnh gốc", use_container_width=True)
            with col2:
                st.image(result4, caption=f"Kết quả: {option4}", use_container_width=True)

    # Chương 9
    elif chapter == "Chương 9: Nhận dạng đối tượng":
        st.markdown("### 📍 Chương 9: Nhận dạng đối tượng")
        uploaded_file_9 = st.file_uploader("📤 Tải ảnh (Chương 9)", type=["jpg","jpeg","png","tif"], key="c9")
        if uploaded_file_9:
            image9 = Image.open(uploaded_file_9).convert("L")
            img9 = np.array(image9)

            method9 = st.selectbox("🔧 Phương pháp Chương 9:", [
                "Erosion", "Dilation", "Boundary", "Contour",
                "ConvexHull", "DefectDetect", "HoleFill",
                "ConnectedComponents", "RemoveSmallRice"
            ], key="opt9")

            if method9 == "Erosion":
                result9 = c9.Erosion(img9)
            elif method9 == "Dilation":
                result9 = c9.Dilation(img9)
            elif method9 == "Boundary":
                result9 = c9.Boundary(img9)
            elif method9 == "Contour":
                result9 = c9.Contour(img9)
            elif method9 == "ConvexHull":
                result9 = c9.ConvexHull(img9)
            elif method9 == "DefectDetect":
                result9 = c9.DefectDetect(img9)
            elif method9 == "HoleFill":
                result9 = c9.HoleFill(img9)
            elif method9 == "ConnectedComponents":
                result9 = c9.ConnectedComponents(img9)
            else:  # RemoveSmallRice
                result9 = c9.RemoveSmallRice(img9)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image9, caption="Ảnh gốc", use_container_width=True)
            with col2:
                st.image(result9, caption=f"Kết quả: {method9}", use_container_width=True)

elif menu == "Mở rộng thêm (tuỳ chọn)":
    st.subheader("🧪 Phần mở rộng")
    st.write("💡 Demo thêm: nhận dạng ảnh 3D, xử lý ảnh nâng cao...")