import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

def plot_confusion_matrix(y_true, y_pred, classes, exp_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {exp_name}')
    plt.tight_layout()
    
    os.makedirs("confusion_matrices", exist_ok=True)
    plt.savefig(f'confusion_matrices/{exp_name}_confusion_matrix.png')
    plt.close()