import os
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sign_model_26class.h5")
DATA_DIR = os.path.join(BASE_DIR, "data", "AtoZ_raw")

print(f"[INFO] Evaluating Model: {MODEL_PATH}")
model = load_model(MODEL_PATH, compile=False)

eval_datagen = ImageDataGenerator(validation_split=0.2)

eval_generator = eval_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

predictions = model.predict(eval_generator, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = eval_generator.classes
labels = list(eval_generator.class_indices.keys())

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print("\n" + "=" * 60)
print(f"  Achieved Accuracy     : {acc * 100:.2f}%")
print(f"  Weighted Precision    : {prec * 100:.2f}%")
print(f"  Weighted Recall       : {rec * 100:.2f}%")
print(f"  Weighted F1-Score     : {f1 * 100:.2f}%")
print("=" * 60)
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))