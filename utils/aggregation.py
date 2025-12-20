import numpy as np

def aggregate_patient_predictions(logits):
    """
    Aggregate scan-level logits into patient-level score
    """
    return float(np.mean(logits))
