import matplotlib.pyplot as plt

def plot_cost_history(cost_history):
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, label='Cost Function')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Function History')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.savefig('cost_history.png')

def plot_predictions(y_true, y_pred):
  
    plt.figure(figsize=(10,6))
    plt.scatter(y_true, y_pred, color='blue')

    plt.plot([min(y_true), max(y_true)],
         [min(y_true), max(y_true)], color='red', linestyle='--')

    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")

    plt.title("True vs Predicted")

    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.savefig('predictions.png')

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(residuals)), residuals, color='purple')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Sample Index')
    plt.ylabel('Residual')
    plt.title('Residual Plot')
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.savefig('residuals.png')