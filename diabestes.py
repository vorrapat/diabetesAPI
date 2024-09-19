import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib

print("Starting the script...")

# โหลดข้อมูลจากไฟล์ CSV
print("Reading CSV file...")
df = pd.read_csv('C:/Users/User/Desktop/VorrapatAIAPI/diabetes.csv')  # ใช้พาธที่ถูกต้อง

print("CSV file read successfully.")

# ลบตัวแปร "Diabetes Pedigree Function" ออกจากข้อมูล
df = df.drop('DiabetesPedigreeFunction', axis=1)
print("Dropped DiabetesPedigreeFunction column.")

# แบ่งข้อมูลออกเป็น features (X) และ target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']
print("Separated features and target.")

# แบ่งข้อมูลออกเป็นชุดฝึกสอนและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print("Split data into training and test sets.")

# ปรับขนาดข้อมูลให้มีมาตรฐานเดียวกัน
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Data has been scaled.")

# สร้างโมเดล K-Nearest Neighbours
model = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)
model.fit(X_train, y_train)
print("Model has been trained.")

# บันทึกโมเดล
joblib.dump(model, "Diabetes_Model.pkl")
print("Model has been saved as 'Diabetes_Model.pkl'.")

# บันทึก Scaler เพื่อใช้กับข้อมูลใหม่
joblib.dump(scaler, "Scaler.pkl")
print("Scaler has been saved as 'Scaler.pkl'.")

print("Script finished successfully.")