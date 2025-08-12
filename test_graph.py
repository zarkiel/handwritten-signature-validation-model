import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def generate_validation_report(csv_file_path):
    """
    Loads a CSV file, calculates key validity metrics, and generates
    graphs and a table for a scientific report.
    
    Args:
        csv_file_path (str): The path to the CSV file.
    """
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: The file was not found at the specified path: {csv_file_path}")
        return

    # --- 1. Generate predictions from distances and thresholds ---
    # A prediction is '1' (Accepted) if distance <= threshold.
    # It's '0' (Rejected) if distance > threshold.
    df['Prediction_label'] = np.where(df['Distance'] <= df['Threshold'], 1, 0)
    
    y_true = df['Label']
    y_pred = df['Prediction_label']

    # --- 2. Calculate all key metrics ---
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    far = fp / (fp + tn) # False Acceptance Rate
    frr = fn / (fn + tp) # False Rejection Rate
    
    # Calculate EER from the ROC curve data
    fpr, tpr, thresholds = roc_curve(y_true, 1 - df['Distance'])
    fnr = 1 - tpr
    eer_threshold_index = np.argmin(np.abs(fnr - fpr))
    eer = (fpr[eer_threshold_index] + fnr[eer_threshold_index]) / 2

    # --- 3. Generate and visualize the Confusion Matrix with metrics ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Rejected (0)', 'Accepted (1)'],
                yticklabels=['Genuine (1)', 'Forged (0)'])
    plt.title('Model Confusion Matrix')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    
    # Add text annotations for key metrics
    plt.text(1.5, -0.2, f"Accuracy: {accuracy:.4f}", ha='center', fontsize=12, weight='bold')
    plt.text(1.5, -0.05, f"FAR: {far:.4f}", ha='center', fontsize=10)
    plt.text(1.5, 0.1, f"FRR: {frr:.4f}", ha='center', fontsize=10)
    
    plt.show()
    #plt.savefig('results/confusion_matrix.png')
    #plt.close()

    # --- 4. Generate and visualize the ROC Curve with metrics ---
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.scatter(fpr[eer_threshold_index], tpr[eer_threshold_index], marker='o', color='red', s=100,
                label=f'EER Point = {eer:.4f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    
    plt.title(f'Model ROC Curve\nAccuracy: {accuracy:.4f}, EER: {eer:.4f}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    #plt.savefig('results/roc_curve.png')
    #plt.close()

    # --- 5. Generate and display the key metrics table ---
    data = {'Metric': ['Accuracy', 'FAR', 'FRR', 'EER'],
            'Value': [f'{accuracy:.4f}', f'{far:.4f}', f'{frr:.4f}', f'{eer:.4f}']}
    
    df_metrics = pd.DataFrame(data)
    
    print("\n" + "-"*50)
    print("Key Metrics Table:")
    print(df_metrics.to_string(index=False))
    print("-" * 50)
    
if __name__ == "__main__":
    file_name = "test_results.csv"
    generate_validation_report(file_name)