# **Sign-Language-to-Sentences**

A real-time sign language recognition system that converts hand gestures into meaningful sentences using deep learning and computer vision.

## **Features**
✅ Real-time sign language recognition using **MediaPipe** and **LSTM-based Deep Learning Model**  
✅ Converts sign gestures into readable text  
✅ Live video feed with overlaid predictions  
✅ Supports both **live camera input** and **pre-recorded videos**  
✅ Flask-based web interface for easy interaction  

---

## **Tech Stack**
- **Frontend**: HTML, CSS, JavaScript  
- **Backend**: Flask (Python)  
- **Deep Learning**: TensorFlow, Keras, LSTMs  
- **Computer Vision**: OpenCV, MediaPipe  
- **Database (if needed)**: SQLite / Firebase  

---

## **Setup Instructions**

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/Sign-Language-to-Sentences.git
cd Sign-Language-to-Sentences
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Download Model Weights**
Place the trained model weights (`action.h5`) in the `model/` directory.

### **4️⃣ Run the Flask App**
```bash
python app.py
```
Then, open [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.

---

## **Website Screenshots**

### **Home Page**
![Home Page](![Uploading image.png…]())


### **About Page**
![About Page](![image](https://github.com/user-attachments/assets/ef8df46d-4751-49bf-9d13-dee6329b41fb)
)

### **Services Page**
![Services Page](![image](https://github.com/user-attachments/assets/03da3e23-e0b7-4fe6-9e95-80e3ae8e6d2b)
)

### **Live Video Feed**
![Live Video Feed](![image](https://github.com/user-attachments/assets/236e60c5-6051-4835-87eb-de8975561fac)
)

### **Upload Video**
![Upload Video](![image](https://github.com/user-attachments/assets/5145baaa-6a64-4b20-9a60-e2e29c159967)
)
)

### **Contact Page**
![Contact Page](![{5790F8F0-F0D5-4B7D-9963-85905902FA39}](https://github.com/user-attachments/assets/22cf3ade-6ca0-4d8c-9577-9d0072476521)
)


## **Project Structure**
```
Sign-Language-to-Sentences/
│── model/                      # Pre-trained model weights (action.h5)
│── static/                     # Static assets (CSS, JS, Images)
│── templates/                  # HTML Templates (Flask views)
│── uploads/                    # Uploaded video files
│── app.py                       # Main Flask application
│── requirements.txt             # List of dependencies
│── README.md                    # Project documentation
```

---

## **Usage**
- **Live Video Feed**: Click "Live Camera Feed" to recognize gestures in real-time.
- **Upload Video**: Upload a recorded video for sign language detection.
- **Text Output**: Recognized signs are displayed as continuous text.

---



## **Contributors**
- **Kshitiz Kumar**
- **Sanyam Sankhla**
- **Anshul Dewangan**
- **Akshat Arora**
- **Aranyak Banerjee**


---

## **License**
📜 MIT License – Free to use and modify.  
