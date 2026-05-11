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
    plt.savefig("outputs/cost_history.png")
    plt.show()
    plt.close()

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
    plt.savefig("outputs/predictions.png")
    plt.show()
    plt.close()
    
def plot_residuals(y_true, y_pred):

    residuals = y_true - y_pred

    plt.figure(figsize=(10, 6))

    plt.scatter(y_pred, residuals)

    plt.axhline(0, linestyle='--')

    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")

    plt.title("Residual Plot")

    plt.grid()
    plt.tight_layout()

    plt.savefig("outputs/residuals.png")
    plt.show()
    plt.close()