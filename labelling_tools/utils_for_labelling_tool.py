import numpy as np
import torch
import os
from skimage.io import imread
import matplotlib.pyplot as plt
import matplotlib

data_dir = "/vol/biomedic3/data/EMBED/images/png/1024x768/"


def get_images(img_paths):
    """
    returns a array of images

            Parameters:
                    img_paths (array): List of image paths
            Returns:
                    images (array): List of torch arrays of images
    """
    images = []
    for img_path in img_paths:
        full_img_path = os.path.join(data_dir, img_path)
        image = imread(full_img_path).astype(np.float32)
        image = torch.from_numpy(image)
        images.append(image)
    return images


def plot_images(
    multilabel,
    image_set,
    markers,
    label_index=0,
    batch=0,
    number=20,
    width=768,
    height=1024,
    n_samples_row=4,
):
    plt.close()
    image_batch = image_set[batch * number : (batch + 1) * number]  # noqa
    markers_batch = markers[batch * number : (batch + 1) * number]  # noqa
    indices = image_batch.keys()

    def on_click(event):
        def get_index(x, y):
            col = x // width
            row = y // height
            index = row * n_samples_row + col
            return index

        if event.inaxes != ax:
            return

        # Get the coordinates of the click
        x, y = int(event.xdata), int(event.ydata)
        index = get_index(x, y)
        data_index = indices[index]
        if data_index in markers_batch:
            if multilabel:
                increment = True if markers[data_index][label_index] == 0 else False
                markers[data_index][label_index] += 1 if increment else -1
            else:
                increment = True if markers[data_index] != 7 else False
                markers[data_index] += 1 if increment else -7
        redraw()

    def redraw():
        ax.clear()
        ax.imshow(image, cmap=matplotlib.cm.gray, clim=None)
        # label images
        for index in range(len(indices)):
            row = index // n_samples_row
            col = index % n_samples_row
            data_index = indices[index]
            ax.text(
                (col) * width,
                (row) * height,
                str(markers[data_index]),
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )
        ax.axis("off")
        fig.patch.set_facecolor("black")
        fig.canvas.draw()

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    image_height, image_width = height, width
    blank_image = np.zeros((image_height, image_width), dtype=np.uint8)
    n_rows = (number - 1) // n_samples_row + 1
    images_row = []
    for current_row in range(n_rows):
        tmp_row_images = list(
            image_batch[
                current_row * n_samples_row : (current_row + 1) * n_samples_row
            ]  # noqa
        )
        while len(tmp_row_images) < n_samples_row:
            tmp_row_images.append(blank_image)
        images_row.append(np.concatenate(list(tmp_row_images), axis=1))

    image = np.concatenate(images_row, axis=0)
    redraw()
    # Add click event
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
