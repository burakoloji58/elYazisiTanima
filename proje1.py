import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
import os

# ---  MNIST veri seti 
(x_data, y_data), (x_test, y_test) = mnist.load_data()

x_data = x_data.astype('float32') / 255.0
x_data = x_data.reshape((x_data.shape[0], 28, 28, 1))
y_data = to_categorical(y_data, 10)

# ---  Modeli tanımladık (CNN) 
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ---  Cross Validation ile eğittik 
model_save_path = 'handwritten_digit_model.h5'

if os.path.exists(model_save_path):
    print("Önceden kaydedilmiş model yüklendi.")
    model = tf.keras.models.load_model(model_save_path)
else:
    print("Cross-validation ile model eğitiliyor...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores = []

    fold_no = 1
    for train_idx, val_idx in kfold.split(x_data):
        print(f"Fold {fold_no} başlıyor...")
        x_train, x_val = x_data[train_idx], x_data[val_idx]
        y_train, y_val = y_data[train_idx], y_data[val_idx]

        model = create_model()
        model.fit(x_train, y_train, epochs=3, batch_size=64, verbose=0, validation_data=(x_val, y_val))

        score = model.evaluate(x_val, y_val, verbose=0)
        acc_scores.append(score[1])
        print(f"Fold {fold_no} doğruluk: {score[1]*100:.2f}%")
        fold_no += 1

    print(f"Ortalama doğruluk: {np.mean(acc_scores) * 100:.2f}%")

    #  modeli kaydettik
    model.save(model_save_path)
    print("Model kaydedildi.")

# --- OpenCV kullanarak tahmin yaptık 
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        roi = thresh[y:y+h, x:x+w]

        resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        resized = resized.astype('float32') / 255.0
        resized = resized.reshape(1, 28, 28, 1)

        prediction = model.predict(resized)
        predicted_class = np.argmax(prediction)

        cv2.putText(frame, f"Prediction: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    combined = np.hstack((frame, cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)))
    cv2.imshow('Handwritten Digit Recognition - Color & BW', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
