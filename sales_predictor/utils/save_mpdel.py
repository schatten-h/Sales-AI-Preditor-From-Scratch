import pickle
import config

config.ALPHA

def save_model(w, b, filepath):

    model_data = {
        "weights": w,
        "bias": b
    }

    with open(filepath, "wb") as f:
        pickle.dump(model_data, f)

def load_model(filepath):

    with open(filepath, "rb") as f:
        model_data = pickle.load(f)

    return (
        model_data["weights"],
        model_data["bias"]
    )

