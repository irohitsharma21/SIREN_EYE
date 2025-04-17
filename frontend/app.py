import streamlit as st
import requests
import tempfile
import os

# Update this with your actual backend Render URL
BACKEND_URL = "https://ambulance-detection-backend.onrender.com"

# Streamlit UI
st.title("🚑 Ambulance Detection System")
st.write("Upload a video to detect an emergency ambulance.")

uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_video:
    # Show the uploaded video
    st.video(uploaded_video)

    if st.button("Detect Ambulance 🚨"):
        st.info("Processing video, please wait...")

        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_video.read())
            temp_file_path = temp_file.name

        try:
            # Send file to backend
            with open(temp_file_path, "rb") as f:
                files = {"video": f}
                response = requests.post(f"{BACKEND_URL}/detect", files=files)

            if response.status_code == 200:
                result = response.json()
                st.success("✅ Detection completed!")
                st.write("### Detection Results")
                st.json(result)
            else:
                st.error(f"Detection failed: {response.text}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

        finally:
            # Clean up temp file
            os.remove(temp_file_path)
