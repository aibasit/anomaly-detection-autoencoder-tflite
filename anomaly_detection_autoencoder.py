"""
==============================================================================
MLES Assignment 3: Anomaly Detection using AutoEncoders
==============================================================================
Dataset: KDDCup99
Objective: Train an AutoEncoder on normal network traffic, detect anomalies
           using reconstruction error, then quantize the model and compare
           performance, size, and speed.
==============================================================================
"""

# ============================================================================
# 0. IMPORTS AND CONFIGURATION
# ============================================================================

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Create output directories
PLOT_DIR = 'plots'
MODEL_DIR = 'models'
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 70)
print("MLES Assignment 3: Anomaly Detection using AutoEncoders")
print("=" * 70)
print("TensorFlow Version:", tf.__version__)
print("NumPy Version:", np.__version__)
print("Pandas Version:", pd.__version__)
print()

# ============================================================================
# 1. LOAD AND UNDERSTAND DATASET
# ============================================================================

print("=" * 70)
print("STEP 1: Load and Understand Dataset")
print("=" * 70)

df = pd.read_csv('KDDCup99.csv')

print("\n[INFO] Dataset Shape:", df.shape)
print("   Rows:", f"{df.shape[0]:,}")
print("   Columns:", df.shape[1])

print("\n[INFO] Column Names (%d):" % len(df.columns))
for i, col in enumerate(df.columns, 1):
    print("   %2d. %s" % (i, col))

print("\n[INFO] Dataset Info:")
print(df.info())

print("\n[INFO] Statistical Summary (first 5 numeric columns):")
print(df.describe().iloc[:, :5])

# Identify categorical and numerical features
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

print("\n[INFO] Categorical Features (%d): %s" % (len(categorical_cols), categorical_cols))
print("[INFO] Numerical Features (%d columns)" % len(numerical_cols))

# Label distribution
print("\n[INFO] Label Distribution:")
label_counts = df['label'].value_counts()
for label, count in label_counts.items():
    pct = count / len(df) * 100
    print("   %-20s: %8s (%5.2f%%)" % (label, f"{count:,}", pct))

normal_count = df[df['label'] == 'normal'].shape[0]
attack_count = df[df['label'] != 'normal'].shape[0]
print("\n   [+] Normal: %s (%.2f%%)" % (f"{normal_count:,}", normal_count/len(df)*100))
print("   [-] Attack: %s (%.2f%%)" % (f"{attack_count:,}", attack_count/len(df)*100))

# ============================================================================
# 2. PREPROCESSING
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2: Preprocessing")
print("=" * 70)

# 2a. Convert label into binary target
# normal -> 0, all attacks -> 1
df['target'] = (df['label'] != 'normal').astype(int)

print("\n[OK] Created binary 'target' column:")
print("   Normal (0):", f"{(df['target'] == 0).sum():,}")
print("   Anomaly (1):", f"{(df['target'] == 1).sum():,}")

# 2b. Separate features (X) and target (y)
# Drop both 'label' (original string) and 'target' from features
y = df['target'].values
X = df.drop(columns=['label', 'target'])

print("\n[INFO] Features shape before encoding:", X.shape)

# 2c. One-hot encoding for categorical columns
categorical_features = ['protocol_type', 'service', 'flag']
print("\n[INFO] Applying One-Hot Encoding on:", categorical_features)

for col in categorical_features:
    print("   %s: %d unique values" % (col, X[col].nunique()))

X = pd.get_dummies(X, columns=categorical_features, dtype=float)

print("   [OK] Features shape after encoding:", X.shape)

# 2d. Normalize all features using MinMaxScaler
print("\n[INFO] Applying MinMaxScaler normalization...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("   [OK] Feature range: [%.4f, %.4f]" % (X_scaled.min().min(), X_scaled.max().max()))
print("   [OK] Final feature dimensions:", X_scaled.shape)

# ============================================================================
# 3. DATA SPLITTING
# ============================================================================

print("\n" + "=" * 70)
print("STEP 3: Data Splitting")
print("=" * 70)

# 3a. Train-Test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=SEED, stratify=y
)

print("\n[INFO] Train-Test Split (80/20):")
print("   Training set: %s samples" % f"{X_train.shape[0]:,}")
print("   Test set:     %s samples" % f"{X_test.shape[0]:,}")
print("   Train normal: %s | Train anomaly: %s" % (f"{(y_train == 0).sum():,}", f"{(y_train == 1).sum():,}"))
print("   Test normal:  %s | Test anomaly:  %s" % (f"{(y_test == 0).sum():,}", f"{(y_test == 1).sum():,}"))

# 3b. Keep only NORMAL samples for training the AutoEncoder
X_train_normal = X_train[y_train == 0]

print("\n[INFO] AutoEncoder Training Data (Normal only):")
print("   Samples:", f"{X_train_normal.shape[0]:,}")
print("   Features:", X_train_normal.shape[1])

# Convert to numpy arrays
X_train_normal = X_train_normal.values.astype('float32')
X_test_np = X_test.values.astype('float32')

input_dim = X_train_normal.shape[1]
print("\n   Input dimension for AutoEncoder:", input_dim)

# ============================================================================
# 4. BUILD AUTOENCODER MODEL
# ============================================================================

print("\n" + "=" * 70)
print("STEP 4: Build AutoEncoder Model")
print("=" * 70)

# Architecture: Input -> 128 -> 64 -> 32 -> 16 (bottleneck) -> 32 -> 64 -> 128 -> Output

input_layer = Input(shape=(input_dim,), name='input')

# Encoder
encoded = Dense(128, activation='relu', name='encoder_1')(input_layer)
encoded = Dense(64, activation='relu', name='encoder_2')(encoded)
encoded = Dense(32, activation='relu', name='encoder_3')(encoded)
bottleneck = Dense(16, activation='relu', name='bottleneck')(encoded)

# Decoder
decoded = Dense(32, activation='relu', name='decoder_1')(bottleneck)
decoded = Dense(64, activation='relu', name='decoder_2')(decoded)
decoded = Dense(128, activation='relu', name='decoder_3')(decoded)
output_layer = Dense(input_dim, activation='sigmoid', name='output')(decoded)

# Build model
autoencoder = Model(inputs=input_layer, outputs=output_layer, name='AutoEncoder')

# Compile
autoencoder.compile(
    optimizer='adam',
    loss='mse'
)

print("\n[INFO] AutoEncoder Architecture:")
autoencoder.summary()

print("\n   Optimizer: Adam")
print("   Loss: Mean Squared Error (MSE)")
print("   Hidden Activation: ReLU")
print("   Output Activation: Sigmoid")

# ============================================================================
# 5. TRAIN MODEL
# ============================================================================

print("\n" + "=" * 70)
print("STEP 5: Train AutoEncoder Model")
print("=" * 70)

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, 'best_autoencoder.keras'),
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

print("\n[START] Training AutoEncoder (on normal data only)...")
print("   Epochs: 50 (with EarlyStopping, patience=5)")
print("   Batch size: 256")
print("   Validation split: 10%")

history = autoencoder.fit(
    X_train_normal, X_train_normal,  # Input = Output (reconstruction)
    epochs=50,
    batch_size=256,
    validation_split=0.1,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

print("\n[OK] Training completed!")
print("   Final training loss:   %.6f" % history.history['loss'][-1])
print("   Final validation loss: %.6f" % history.history['val_loss'][-1])
print("   Total epochs trained:  %d" % len(history.history['loss']))

# Save the final model
autoencoder.save(os.path.join(MODEL_DIR, 'autoencoder_final.keras'))
print("   [OK] Model saved to %s/autoencoder_final.keras" % MODEL_DIR)

# ============================================================================
# 6. ANOMALY DETECTION
# ============================================================================

print("\n" + "=" * 70)
print("STEP 6: Anomaly Detection")
print("=" * 70)

# 6a. Reconstruct test data
print("\n[INFO] Reconstructing test data...")
X_test_pred = autoencoder.predict(X_test_np, verbose=0)

# 6b. Compute reconstruction error (MSE per sample)
reconstruction_error_test = np.mean(np.square(X_test_np - X_test_pred), axis=1)

print("   Test reconstruction error stats:")
print("   Mean:   %.6f" % np.mean(reconstruction_error_test))
print("   Std:    %.6f" % np.std(reconstruction_error_test))
print("   Min:    %.6f" % np.min(reconstruction_error_test))
print("   Max:    %.6f" % np.max(reconstruction_error_test))

# 6c. Determine threshold using training reconstruction error
print("\n[INFO] Computing threshold from training data...")
X_train_pred = autoencoder.predict(X_train_normal, verbose=0)
reconstruction_error_train = np.mean(np.square(X_train_normal - X_train_pred), axis=1)

# Method 1: mean + 3 * std
threshold_3sigma = np.mean(reconstruction_error_train) + 3 * np.std(reconstruction_error_train)

# Method 2: 95th percentile
threshold_95pct = np.percentile(reconstruction_error_train, 95)

print("   Training reconstruction error stats:")
print("   Mean: %.6f" % np.mean(reconstruction_error_train))
print("   Std:  %.6f" % np.std(reconstruction_error_train))
print("\n   Threshold (mean + 3*std):     %.6f" % threshold_3sigma)
print("   Threshold (95th percentile):  %.6f" % threshold_95pct)

# Use mean + 3*std as primary threshold
threshold = threshold_3sigma
print("\n   [OK] Selected Threshold: %.6f (mean + 3*std)" % threshold)

# 6d. Classify: error > threshold -> anomaly (1), error <= threshold -> normal (0)
y_pred = (reconstruction_error_test > threshold).astype(int)

print("\n   Predictions:")
print("   Predicted Normal:  %s" % f"{(y_pred == 0).sum():,}")
print("   Predicted Anomaly: %s" % f"{(y_pred == 1).sum():,}")

# ============================================================================
# 7. EVALUATION
# ============================================================================

print("\n" + "=" * 70)
print("STEP 7: Evaluation - Baseline AutoEncoder")
print("=" * 70)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, reconstruction_error_test)
cm = confusion_matrix(y_test, y_pred)

print("\n[RESULTS] Baseline AutoEncoder Performance:")
print("   Accuracy:   %.4f (%.2f%%)" % (accuracy, accuracy*100))
print("   Precision:  %.4f" % precision)
print("   Recall:     %.4f" % recall)
print("   F1-Score:   %.4f" % f1)
print("   ROC-AUC:    %.4f" % roc_auc)

print("\n[INFO] Confusion Matrix:")
print("   %-15s Pred Normal   Pred Anomaly" % "")
print("   %-15s  %8s      %8s" % ("Actual Normal", f"{cm[0][0]:,}", f"{cm[0][1]:,}"))
print("   %-15s  %8s      %8s" % ("Actual Anomaly", f"{cm[1][0]:,}", f"{cm[1][1]:,}"))

print("\n[INFO] Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))

# Store baseline results for comparison
baseline_results = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'ROC-AUC': roc_auc
}

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 70)
print("STEP 8: Visualizations")
print("=" * 70)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# --- 8a. Training vs Validation Loss Curve ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax.set_title('AutoEncoder Training vs Validation Loss', fontsize=16, fontweight='bold')
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Mean Squared Error (MSE)', fontsize=13)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '01_training_validation_loss.png'), dpi=150)
plt.close()
print("   [OK] Saved: plots/01_training_validation_loss.png")

# --- 8b. Histogram of Reconstruction Error (Normal vs Anomaly) ---
error_normal = reconstruction_error_test[y_test == 0]
error_anomaly = reconstruction_error_test[y_test == 1]

fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(error_normal, bins=100, alpha=0.7, label='Normal', color='#2ecc71', density=True)
ax.hist(error_anomaly, bins=100, alpha=0.7, label='Anomaly', color='#e74c3c', density=True)
ax.axvline(x=threshold, color='#f39c12', linestyle='--', linewidth=2,
           label='Threshold (%.4f)' % threshold)
ax.set_title('Reconstruction Error Distribution', fontsize=16, fontweight='bold')
ax.set_xlabel('Reconstruction Error (MSE)', fontsize=13)
ax.set_ylabel('Density', fontsize=13)
ax.legend(fontsize=12)
ax.set_xlim(0, min(np.percentile(reconstruction_error_test, 99.5), 0.5))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '02_reconstruction_error_distribution.png'), dpi=150)
plt.close()
print("   [OK] Saved: plots/02_reconstruction_error_distribution.png")

# --- 8c. Confusion Matrix Heatmap ---
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=',', cmap='Blues',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'],
            ax=ax, annot_kws={'size': 14})
ax.set_title('Confusion Matrix - Baseline AutoEncoder', fontsize=16, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=13)
ax.set_ylabel('Actual', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '03_confusion_matrix_baseline.png'), dpi=150)
plt.close()
print("   [OK] Saved: plots/03_confusion_matrix_baseline.png")

# --- 8d. ROC Curve ---
fpr, tpr, thresholds_roc = roc_curve(y_test, reconstruction_error_test)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(fpr, tpr, color='#3498db', linewidth=2, label='ROC Curve (AUC = %.4f)' % roc_auc)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
ax.set_title('ROC Curve - Baseline AutoEncoder', fontsize=16, fontweight='bold')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '04_roc_curve_baseline.png'), dpi=150)
plt.close()
print("   [OK] Saved: plots/04_roc_curve_baseline.png")

# ============================================================================
# 9. QUANTIZATION - CONVERT MODEL TO TFLITE
# ============================================================================

print("\n" + "=" * 70)
print("STEP 9: Model Quantization (TFLite Conversion)")
print("=" * 70)

# Convert to TFLite with default optimizations (Dynamic Range Quantization)
print("\n[INFO] Converting Keras model to TFLite with quantization...")

converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

print("   [OK] Conversion successful!")

# ============================================================================
# 10. SAVE QUANTIZED MODEL
# ============================================================================

print("\n" + "=" * 70)
print("STEP 10: Save Quantized Model")
print("=" * 70)

tflite_model_path = os.path.join(MODEL_DIR, 'autoencoder_quantized.tflite')
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print("   [OK] Quantized model saved to:", tflite_model_path)

# ============================================================================
# 11. COMPARE MODEL SIZE
# ============================================================================

print("\n" + "=" * 70)
print("STEP 11: Compare Model Size")
print("=" * 70)

# Original model size
original_model_path = os.path.join(MODEL_DIR, 'autoencoder_final.keras')
original_size = os.path.getsize(original_model_path)

# Quantized model size
quantized_size = os.path.getsize(tflite_model_path)

# Compute reduction
size_reduction = (1 - quantized_size / original_size) * 100

print("\n[RESULTS] Model Size Comparison:")
print("   Original model:   %10s bytes (%.2f KB)" % (f"{original_size:,}", original_size/1024))
print("   Quantized model:  %10s bytes (%.2f KB)" % (f"{quantized_size:,}", quantized_size/1024))
print("   Size reduction:   %.2f%%" % size_reduction)

# ============================================================================
# 12. MEASURE INFERENCE SPEED
# ============================================================================

print("\n" + "=" * 70)
print("STEP 12: Measure Inference Speed")
print("=" * 70)

NUM_INFERENCE_SAMPLES = 1000
test_samples = X_test_np[:NUM_INFERENCE_SAMPLES]

# --- 12a. Keras Model Inference Speed ---
print("\n[INFO] Measuring Keras model inference time (%d samples)..." % NUM_INFERENCE_SAMPLES)

# Warm up
_ = autoencoder.predict(test_samples[:10], verbose=0)

keras_times = []
for i in range(5):  # 5 runs for averaging
    start = time.perf_counter()
    _ = autoencoder.predict(test_samples, verbose=0)
    end = time.perf_counter()
    keras_times.append(end - start)

keras_avg_time = np.mean(keras_times)
keras_per_sample = keras_avg_time / NUM_INFERENCE_SAMPLES * 1000  # in ms

print("   Keras avg total time:     %.2f ms" % (keras_avg_time*1000))
print("   Keras avg per sample:     %.4f ms" % keras_per_sample)

# --- 12b. TFLite Model Inference Speed ---
print("\n[INFO] Measuring TFLite model inference time (%d samples)..." % NUM_INFERENCE_SAMPLES)

# Setup TFLite interpreter
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Warm up
for j in range(10):
    sample = test_samples[j:j+1].astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()

tflite_times = []
for i in range(5):  # 5 runs for averaging
    start = time.perf_counter()
    for j in range(NUM_INFERENCE_SAMPLES):
        sample = test_samples[j:j+1].astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
    end = time.perf_counter()
    tflite_times.append(end - start)

tflite_avg_time = np.mean(tflite_times)
tflite_per_sample = tflite_avg_time / NUM_INFERENCE_SAMPLES * 1000  # in ms

print("   TFLite avg total time:    %.2f ms" % (tflite_avg_time*1000))
print("   TFLite avg per sample:    %.4f ms" % tflite_per_sample)

speedup = keras_avg_time / tflite_avg_time if tflite_avg_time > 0 else float('inf')
print("\n   [RESULT] Speedup factor: %.2fx" % speedup)

# ============================================================================
# 13. EVALUATE QUANTIZED MODEL
# ============================================================================

print("\n" + "=" * 70)
print("STEP 13: Evaluate Quantized Model")
print("=" * 70)

# Run inference using TFLite interpreter on full test set
print("\n[INFO] Running TFLite inference on full test set (%s samples)..." % f"{X_test_np.shape[0]:,}")

tflite_predictions = np.zeros_like(X_test_np)
for i in range(X_test_np.shape[0]):
    sample = X_test_np[i:i+1].astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()
    tflite_predictions[i] = interpreter.get_tensor(output_details[0]['index'])[0]
    
    if (i + 1) % 20000 == 0:
        print("   Processed %s/%s samples..." % (f"{i+1:,}", f"{X_test_np.shape[0]:,}"))

print("   [OK] Inference complete!")

# Compute reconstruction error for TFLite
reconstruction_error_tflite = np.mean(np.square(X_test_np - tflite_predictions), axis=1)

# Use same threshold
y_pred_tflite = (reconstruction_error_tflite > threshold).astype(int)

# Compute metrics
accuracy_q = accuracy_score(y_test, y_pred_tflite)
precision_q = precision_score(y_test, y_pred_tflite)
recall_q = recall_score(y_test, y_pred_tflite)
f1_q = f1_score(y_test, y_pred_tflite)
roc_auc_q = roc_auc_score(y_test, reconstruction_error_tflite)
cm_q = confusion_matrix(y_test, y_pred_tflite)

print("\n[RESULTS] Quantized AutoEncoder Performance:")
print("   Accuracy:   %.4f (%.2f%%)" % (accuracy_q, accuracy_q*100))
print("   Precision:  %.4f" % precision_q)
print("   Recall:     %.4f" % recall_q)
print("   F1-Score:   %.4f" % f1_q)
print("   ROC-AUC:    %.4f" % roc_auc_q)

print("\n[INFO] Confusion Matrix (Quantized):")
print("   %-15s Pred Normal   Pred Anomaly" % "")
print("   %-15s  %8s      %8s" % ("Actual Normal", f"{cm_q[0][0]:,}", f"{cm_q[0][1]:,}"))
print("   %-15s  %8s      %8s" % ("Actual Anomaly", f"{cm_q[1][0]:,}", f"{cm_q[1][1]:,}"))

print("\n[INFO] Classification Report (Quantized):")
print(classification_report(y_test, y_pred_tflite, target_names=['Normal', 'Anomaly']))

quantized_results = {
    'Accuracy': accuracy_q,
    'Precision': precision_q,
    'Recall': recall_q,
    'F1-Score': f1_q,
    'ROC-AUC': roc_auc_q
}

# ============================================================================
# ADDITIONAL VISUALIZATIONS FOR QUANTIZED MODEL
# ============================================================================

print("\n" + "=" * 70)
print("Additional Visualizations")
print("=" * 70)

# --- Confusion Matrix for Quantized Model ---
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_q, annot=True, fmt=',', cmap='Oranges',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'],
            ax=ax, annot_kws={'size': 14})
ax.set_title('Confusion Matrix - Quantized AutoEncoder', fontsize=16, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=13)
ax.set_ylabel('Actual', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '05_confusion_matrix_quantized.png'), dpi=150)
plt.close()
print("   [OK] Saved: plots/05_confusion_matrix_quantized.png")

# --- ROC Curve Comparison ---
fpr_q, tpr_q, _ = roc_curve(y_test, reconstruction_error_tflite)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(fpr, tpr, color='#3498db', linewidth=2, label='Baseline (AUC = %.4f)' % roc_auc)
ax.plot(fpr_q, tpr_q, color='#e74c3c', linewidth=2, linestyle='--',
        label='Quantized (AUC = %.4f)' % roc_auc_q)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
ax.set_title('ROC Curve Comparison', fontsize=16, fontweight='bold')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '06_roc_curve_comparison.png'), dpi=150)
plt.close()
print("   [OK] Saved: plots/06_roc_curve_comparison.png")

# --- Model Size Comparison Bar Chart ---
fig, ax = plt.subplots(figsize=(8, 6))
model_names = ['Baseline\nAutoEncoder', 'Quantized\nAutoEncoder']
sizes_kb = [original_size / 1024, quantized_size / 1024]
colors = ['#3498db', '#e74c3c']
bars = ax.bar(model_names, sizes_kb, color=colors, width=0.5, edgecolor='black', linewidth=0.5)
for bar, size in zip(bars, sizes_kb):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(sizes_kb)*0.02,
            '%.1f KB' % size, ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_title('Model Size Comparison (%.1f%% Reduction)' % size_reduction, fontsize=16, fontweight='bold')
ax.set_ylabel('Model Size (KB)', fontsize=13)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '07_model_size_comparison.png'), dpi=150)
plt.close()
print("   [OK] Saved: plots/07_model_size_comparison.png")

# --- Inference Speed Comparison Bar Chart ---
fig, ax = plt.subplots(figsize=(8, 6))
model_names = ['Keras\nModel', 'TFLite\n(Quantized)']
times = [keras_per_sample, tflite_per_sample]
colors = ['#3498db', '#e74c3c']
bars = ax.bar(model_names, times, color=colors, width=0.5, edgecolor='black', linewidth=0.5)
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(times)*0.02,
            '%.4f ms' % t, ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_title('Inference Speed Comparison (%.2fx Speedup)' % speedup, fontsize=16, fontweight='bold')
ax.set_ylabel('Time per Sample (ms)', fontsize=13)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '08_inference_speed_comparison.png'), dpi=150)
plt.close()
print("   [OK] Saved: plots/08_inference_speed_comparison.png")

# --- Reconstruction Error Distribution Comparison ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Baseline
ax = axes[0]
ax.hist(reconstruction_error_test[y_test == 0], bins=100, alpha=0.7, label='Normal', color='#2ecc71', density=True)
ax.hist(reconstruction_error_test[y_test == 1], bins=100, alpha=0.7, label='Anomaly', color='#e74c3c', density=True)
ax.axvline(x=threshold, color='#f39c12', linestyle='--', linewidth=2, label='Threshold')
ax.set_title('Baseline AutoEncoder', fontsize=14, fontweight='bold')
ax.set_xlabel('Reconstruction Error', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_xlim(0, min(np.percentile(reconstruction_error_test, 99.5), 0.5))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Quantized
ax = axes[1]
ax.hist(reconstruction_error_tflite[y_test == 0], bins=100, alpha=0.7, label='Normal', color='#2ecc71', density=True)
ax.hist(reconstruction_error_tflite[y_test == 1], bins=100, alpha=0.7, label='Anomaly', color='#e74c3c', density=True)
ax.axvline(x=threshold, color='#f39c12', linestyle='--', linewidth=2, label='Threshold')
ax.set_title('Quantized AutoEncoder', fontsize=14, fontweight='bold')
ax.set_xlabel('Reconstruction Error', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_xlim(0, min(np.percentile(reconstruction_error_tflite, 99.5), 0.5))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle('Reconstruction Error Distribution Comparison', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '09_error_distribution_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: plots/09_error_distribution_comparison.png")

# ============================================================================
# FINAL COMPARISON TABLE
# ============================================================================

print("\n" + "=" * 70)
print("FINAL RESULTS: Comparison Table")
print("=" * 70)

print("\n" + "=" * 70)
print("%-22s %15s %15s %12s" % ("Metric", "Baseline AE", "Quantized AE", "Difference"))
print("=" * 70)
print("%-22s %15.4f %15.4f %+12.4f" % ("Accuracy", baseline_results['Accuracy'], quantized_results['Accuracy'], quantized_results['Accuracy']-baseline_results['Accuracy']))
print("%-22s %15.4f %15.4f %+12.4f" % ("Precision", baseline_results['Precision'], quantized_results['Precision'], quantized_results['Precision']-baseline_results['Precision']))
print("%-22s %15.4f %15.4f %+12.4f" % ("Recall", baseline_results['Recall'], quantized_results['Recall'], quantized_results['Recall']-baseline_results['Recall']))
print("%-22s %15.4f %15.4f %+12.4f" % ("F1-Score", baseline_results['F1-Score'], quantized_results['F1-Score'], quantized_results['F1-Score']-baseline_results['F1-Score']))
print("%-22s %15.4f %15.4f %+12.4f" % ("ROC-AUC", baseline_results['ROC-AUC'], quantized_results['ROC-AUC'], quantized_results['ROC-AUC']-baseline_results['ROC-AUC']))
print("%-22s %15.2f %15.2f %+11.1f%%" % ("Model Size (KB)", original_size/1024, quantized_size/1024, -size_reduction))
print("%-22s %15.4f %15.4f %11.2fx" % ("Inf. Time (ms/sample)", keras_per_sample, tflite_per_sample, speedup))
print("=" * 70)

# Save results table to CSV
results_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC',
               'Model Size (KB)', 'Inference Time (ms/sample)'],
    'Baseline AE': [
        baseline_results['Accuracy'], baseline_results['Precision'],
        baseline_results['Recall'], baseline_results['F1-Score'],
        baseline_results['ROC-AUC'],
        original_size / 1024, keras_per_sample
    ],
    'Quantized AE': [
        quantized_results['Accuracy'], quantized_results['Precision'],
        quantized_results['Recall'], quantized_results['F1-Score'],
        quantized_results['ROC-AUC'],
        quantized_size / 1024, tflite_per_sample
    ]
})
results_df.to_csv('comparison_results.csv', index=False)
print("\n[OK] Results saved to: comparison_results.csv")

# ============================================================================
# BONUS: THRESHOLD TUNING
# ============================================================================

print("\n" + "=" * 70)
print("BONUS: Threshold Tuning Analysis")
print("=" * 70)

percentiles = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
print("\n%12s %12s %10s %10s %10s %10s" % ("Percentile", "Threshold", "Accuracy", "Precision", "Recall", "F1-Score"))
print("-" * 64)

best_f1 = 0
best_pct = 0
best_thresh = 0

for pct in percentiles:
    t = np.percentile(reconstruction_error_train, pct)
    y_p = (reconstruction_error_test > t).astype(int)
    acc = accuracy_score(y_test, y_p)
    prec = precision_score(y_test, y_p, zero_division=0)
    rec = recall_score(y_test, y_p)
    f1_s = f1_score(y_test, y_p)
    print("%10dth %12.6f %10.4f %10.4f %10.4f %10.4f" % (pct, t, acc, prec, rec, f1_s))
    if f1_s > best_f1:
        best_f1 = f1_s
        best_pct = pct
        best_thresh = t

print("\n[BEST] Threshold: %.6f (at %dth percentile, F1=%.4f)" % (best_thresh, best_pct, best_f1))

# Also test mean + k*std for different k values
print("\n%15s %12s %10s %10s %10s %10s" % ("k (mean+k*std)", "Threshold", "Accuracy", "Precision", "Recall", "F1-Score"))
print("-" * 67)
for k in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
    t = np.mean(reconstruction_error_train) + k * np.std(reconstruction_error_train)
    y_p = (reconstruction_error_test > t).astype(int)
    acc = accuracy_score(y_test, y_p)
    prec = precision_score(y_test, y_p, zero_division=0)
    rec = recall_score(y_test, y_p)
    f1_s = f1_score(y_test, y_p)
    print("%15.1f %12.6f %10.4f %10.4f %10.4f %10.4f" % (k, t, acc, prec, rec, f1_s))

# ============================================================================
# BONUS: DIFFERENT BOTTLENECK SIZES
# ============================================================================

print("\n" + "=" * 70)
print("BONUS: Bottleneck Size Comparison")
print("=" * 70)

bottleneck_sizes = [8, 16, 32]
bottleneck_results = []

for bn_size in bottleneck_sizes:
    print("\n--- Bottleneck size: %d ---" % bn_size)
    
    # Build model with different bottleneck
    inp = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(inp)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(bn_size, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    out = Dense(input_dim, activation='sigmoid')(x)
    
    ae_model = Model(inputs=inp, outputs=out)
    ae_model.compile(optimizer='adam', loss='mse')
    
    # Train
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    ae_model.fit(
        X_train_normal, X_train_normal,
        epochs=50, batch_size=256, validation_split=0.1,
        callbacks=[es], verbose=0
    )
    
    # Evaluate
    pred_train = ae_model.predict(X_train_normal, verbose=0)
    err_train = np.mean(np.square(X_train_normal - pred_train), axis=1)
    t_bn = np.mean(err_train) + 3 * np.std(err_train)
    
    pred_test = ae_model.predict(X_test_np, verbose=0)
    err_test = np.mean(np.square(X_test_np - pred_test), axis=1)
    y_p = (err_test > t_bn).astype(int)
    
    acc = accuracy_score(y_test, y_p)
    prec = precision_score(y_test, y_p)
    rec = recall_score(y_test, y_p)
    f1_s = f1_score(y_test, y_p)
    
    bottleneck_results.append({
        'Bottleneck': bn_size,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1_s
    })
    
    print("   Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f" % (acc, prec, rec, f1_s))
    
    # Clean up
    del ae_model
    tf.keras.backend.clear_session()

# Print comparison
print("\n%12s %10s %10s %10s %10s" % ("Bottleneck", "Accuracy", "Precision", "Recall", "F1-Score"))
print("-" * 52)
for r in bottleneck_results:
    print("%12d %10.4f %10.4f %10.4f %10.4f" % (r['Bottleneck'], r['Accuracy'], r['Precision'], r['Recall'], r['F1-Score']))

# Save bottleneck comparison plot
fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(bottleneck_sizes))
width = 0.2
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

for i, metric in enumerate(metrics):
    values = [r[metric] for r in bottleneck_results]
    ax.bar(x_pos + i * width, values, width, label=metric, color=colors[i])

ax.set_xlabel('Bottleneck Size', fontsize=13)
ax.set_ylabel('Score', fontsize=13)
ax.set_title('Performance vs Bottleneck Size', fontsize=16, fontweight='bold')
ax.set_xticks(x_pos + 1.5 * width)
ax.set_xticklabels(bottleneck_sizes)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '10_bottleneck_comparison.png'), dpi=150)
plt.close()
print("\n   [OK] Saved: plots/10_bottleneck_comparison.png")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("EXECUTION COMPLETE!")
print("=" * 70)

print("""
Generated Files:
   Models:
   - models/autoencoder_final.keras (Original)
   - models/best_autoencoder.keras (Best checkpoint)
   - models/autoencoder_quantized.tflite (Quantized)

   Plots:
   - plots/01_training_validation_loss.png
   - plots/02_reconstruction_error_distribution.png
   - plots/03_confusion_matrix_baseline.png
   - plots/04_roc_curve_baseline.png
   - plots/05_confusion_matrix_quantized.png
   - plots/06_roc_curve_comparison.png
   - plots/07_model_size_comparison.png
   - plots/08_inference_speed_comparison.png
   - plots/09_error_distribution_comparison.png
   - plots/10_bottleneck_comparison.png

   Results:
   - comparison_results.csv

All tasks completed successfully!
""")
