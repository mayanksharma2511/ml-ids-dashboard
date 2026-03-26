import numpy as np

def adaptive_hybrid(rf_preds, gnn_probs, threshold=0.9):
    final_preds = []
    confidences = []

    for i in range(len(rf_preds)):
        gnn_conf = np.max(gnn_probs[i])

        if gnn_conf > threshold:
            pred = np.argmax(gnn_probs[i])
        else:
            pred = rf_preds[i]

        final_preds.append(pred)
        confidences.append(gnn_conf)

    return np.array(final_preds), np.array(confidences)