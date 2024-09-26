# %%
import pandas as pd
from dataset import EMBEDMammoDataModule
from downstream_model import MammoNet
import numpy as np
import torch
from pathlib import Path

torch.multiprocessing.set_sharing_strategy("file_system")

if __name__ == "__main__":
    df_main = pd.read_csv(
        "/vol/biomedic3/data/EMBED/tables/mammo-net-csv/embed-non-negative.csv"
    )
    output_dir = "output/non-negative-baseline/version_0"
    model = MammoNet.load_from_checkpoint(
        output_dir + "/checkpoints/epoch=4-step=26308-v1.ckpt", num_classes=2
    )

    data = EMBEDMammoDataModule(
        csv_file=df_main,
        image_size=(1024, 768),
        target="cancer",
    )

    model = model.cuda()
    model = model.eval()
    preds = []
    labels = []
    img_ids = []
    with torch.no_grad():
        for i, batch in enumerate(data.test_dataloader()):
            if i % 100 == 0:
                print(i)
            preds.append(torch.softmax(model(batch["image"].cuda()), 1).cpu())
            labels.append(batch["label"])
            img_ids.append(batch["image_id"])

    img_ids = np.concatenate(img_ids)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    predictions = pd.DataFrame()
    predictions["image_path"] = img_ids
    predictions["probability"] = preds[:, 1]
    predictions["label"] = labels
    predictions.to_csv(output_dir + "/predictions.csv")
