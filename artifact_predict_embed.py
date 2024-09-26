import numpy as np
import pandas as pd
from pytorch_lightning import seed_everything
from dataset import EMBEDMammoDataModule
from artifact_detector_model import Multilabel_ArtifactDetector, MARKER_NAMES
import torch
from tqdm import tqdm

seed_everything(42, workers=True)
model_dir = "output/artifact-detector/version_0/checkpoints/epoch=6-step=2898.ckpt"
image_size = (512, 384)

base_dataset_csv = pd.read_csv(
    "/vol/biomedic3/data/EMBED/tables/mammo-net-csv/embed-non-negative.csv"
)

num_classes = len(MARKER_NAMES)
for marker in MARKER_NAMES:
    base_dataset_csv[marker] = 0

base_dataset_csv["multilabel_markers"] = base_dataset_csv.apply(
    lambda row: np.array([row[name] for name in MARKER_NAMES]), axis=1
)
data = EMBEDMammoDataModule(
    csv_file=base_dataset_csv,
    image_size=image_size,
    target="artifact",
    batch_size=32,
    split_dataset=False,
    num_workers=22,
)
model = Multilabel_ArtifactDetector.load_from_checkpoint(model_dir)
model = model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.cuda()
predictions = []
image_ids = []
test_loader = data.test_dataloader()
print(len(test_loader))
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        images = batch["image"].to(device)
        ids = batch["image_id"]
        outputs = model(images)
        new_predictions = (outputs > 0.5).int().cpu()
        predictions.append(new_predictions)
        image_ids.append(ids)
        if i % 100 == 0:
            print(i)
predictions = np.concatenate(predictions)
image_ids = np.concatenate(image_ids)
predictions_dataset = pd.DataFrame()
# Update the labelled dataset file with the new labels.
for i in range(len(MARKER_NAMES)):
    predictions_dataset[MARKER_NAMES[i]] = predictions[:, i]

print(image_ids.shape)
print(predictions_dataset.shape)
predictions_dataset["image_path"] = image_ids

predictions_dataset["image_id"] = [
    img_path.split("/")[-1] for img_path in predictions_dataset.image_path.values
]
predictions_dataset.to_csv("predicted_all_embed.csv", index=False)
