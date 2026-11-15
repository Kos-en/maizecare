CLASSES = ["Healthy", "MLND", "MSV", "Blight"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
HEALTHY = "Healthy"
DISEASES = ["MLND", "MSV", "Blight"]

CLF_SIZE = 224   # classifier input
AE_SIZE = 128    # autoencoder input
CACHE_SIZE = 224  # preprocessed cache resolution (AE resizes down from this)
