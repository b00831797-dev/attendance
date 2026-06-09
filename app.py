import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import sys

# Perfected Master Student Roster from your PDF document
STUDENT_ROSTER = sorted([
    "VIDELAINE Come", "ZERMATI--ALVES Mathilde", "de MALHERBE Gauthier", "de TINGUY Philippe",
    "RAHAL Axelle", "RASTOIN Eugenie", "REBUT Paul", "RHOUL Nael", "TASTE Antoine", "TRAN MINH Jean",
    "MOREAU Prune", "MUGNIER-BAJAT Simon", "PARENT du CHATELET Basile", "PAROUTY Achille", 
    "POZZO DI BORGO Henri-Louis", "PREVOST Alberic", "FADDEL Yasmine", "FOATA Paul-Mathieu", 
    "GAYON Louis-Arnaud", "GIHR Arthus", "GISSEROT Albert", "JUST Timothee", "DERAMBURE Venceslas", 
    "DONTHIRI Harsha Vardhan Reddy", "DUCOS-ADER Augustin", "DUHAU Andre", "DUMORTIER Julien", 
    "EPELBAUM Or", "BRUEL Melody", "BUREAU Mathys", "CORTASSE Aurelien", "COTTE Antonin", 
    "DECOOL Hippolyte", "DELMAS Pierre", "ABADIE Pauline", "ABSIL Josephine", "ABTAN Lea", 
    "ASSCHER Andrea", "BARBIER Anne-Sophie", "BENILAN Luc"
])

DB_DIR = "student_database"
if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)

st.set_page_config(page_title="AI Attendance Scanner", page_icon="🎓", layout="wide")
st.title("🎓 Smart Classroom AI Attendance System")
st.write("Match classroom group photos directly against your master student attendance sheet.")

tab1, tab2 = st.tabs(["📸 Match Classroom Picture", "⚙️ Register Student Reference Faces"])

# --- TAB 2: MANAGE & REGISTER REFERENCE FACES ---
with tab2:
    st.header("Register Reference Student Profiles")
    st.write("Upload individual headshots cropped from your attendance PDF sheet to save them into the AI matching index.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_student = st.selectbox("Choose a student name to register:", STUDENT_ROSTER)
        uploaded_face = st.file_uploader(f"Upload reference photo for {selected_student}", type=["jpg", "png", "jpeg"])
    
    if uploaded_face is not None:
        try:
            file_bytes = np.asarray(bytearray(uploaded_face.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img is None:
                st.error("Failed to decode image. Please ensure it's a valid JPG/PNG/JPEG file.")
            else:
                safe_name = selected_student.replace(" ", "_")
                save_path = os.path.join(DB_DIR, f"{safe_name}.jpg")
                cv2.imwrite(save_path, img)
                
                with col2:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Registered Profile Picture: {selected_student}", width=200)
                    st.success(f"Successfully linked face template to: {selected_student}!")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# --- TAB 1: ATTENDANCE MATCHING SCANNER ---
with tab1:
    st.header("Classroom Recognition Engine")
    
    try:
        registered_files = [f for f in os.listdir(DB_DIR) if f.endswith(".jpg")]
        registered_names = [f.replace(".jpg", "").replace("_", " ") for f in registered_files]
    except Exception as e:
        st.error(f"Error reading database: {str(e)}")
        registered_files = []
        registered_names = []
    
    st.info(f"📊 System Status: {len(registered_names)} / {len(STUDENT_ROSTER)} student faces are registered in the reference database.")
    
    uploaded_scene = st.file_uploader("Upload a group photo or a picture of the room:", type=["jpg", "png", "jpeg"])
    
    if uploaded_scene is not None:
        if len(registered_files) == 0:
            st.warning("⚠️ The reference database is empty. Please register student profiles in the configuration tab first.")
        else:
            st.write("🔄 Analyzing visual layers and matching faces...")
            
            try:
                scene_bytes = np.asarray(bytearray(uploaded_scene.read()), dtype=np.uint8)
                scene_img = cv2.imdecode(scene_bytes, cv2.IMREAD_COLOR)
                
                if scene_img is None:
                    st.error("Failed to decode scene image. Please ensure it's a valid JPG/PNG/JPEG file.")
                else:
                    # Stable pixel-matrix template matching routine optimized for cloud environments
                    present_students = []
                    
                    # Resize scene image for faster processing
                    scene_height, scene_width = scene_img.shape[:2]
                    if scene_width > 800:
                        scale = 800 / scene_width
                        scene_img = cv2.resize(scene_img, (int(scene_width * scale), int(scene_height * scale)))
                    
                    scene_gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, (img_file, name) in enumerate(zip(registered_files, registered_names)):
                        try:
                            ref_img = cv2.imread(os.path.join(DB_DIR, img_file), cv2.IMREAD_COLOR)
                            
                            if ref_img is not None:
                                # Resize reference to evaluate consistency structures safely
                                ref_height, ref_width = ref_img.shape[:2]
                                if ref_width > scene_img.shape[1] or ref_height > scene_img.shape[0]:
                                    scale = min(scene_img.shape[1] / ref_width, scene_img.shape[0] / ref_height)
                                    ref_img = cv2.resize(ref_img, (int(ref_width * scale), int(ref_height * scale)))
                                
                                ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
                                
                                # Core Template Matcher algorithm (Native OpenCV, zero compilation required)
                                if ref_gray.shape[0] > 0 and ref_gray.shape[1] > 0:
                                    result = cv2.matchTemplate(scene_gray, ref_gray, cv2.TM_CCOEFF_NORMED)
                                    _, max_val, _, _ = cv2.minMaxLoc(result)
                                    
                                    # If match structural confidence is above 65%, check student as present
                                    if max_val > 0.65:
                                        present_students.append(name)
                        except Exception as e:
                            st.warning(f"Could not process template for {name}: {str(e)}")
                        
                        progress_bar.progress((idx + 1) / len(registered_files))
                        status_text.text(f"Processing: {idx + 1} / {len(registered_files)} faces...")
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success(f"✨ Matching processing completed! Automatically compiled attendance results.")
                    
                    # Generate the live final interactive table matching your request
                    attendance_report = []
                    for student in STUDENT_ROSTER:
                        status = "Present ✅" if student in present_students else "Absent ❌"
                        attendance_report.append({"Student Name": student, "Attendance Status": status})
                    
                    df = pd.DataFrame(attendance_report)
                    st.dataframe(df, use_container_width=True, height=600)
                    
                    # Instant export tool
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Completed Attendance Excel Sheet (.csv)",
                        data=csv_data,
                        file_name="classroom_attendance.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error processing scene image: {str(e)}")
