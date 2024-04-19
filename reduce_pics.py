from pathlib import Path
from PIL import Image
import numpy as np

from source.script import MosaicGenerator


def reduce_resolution(folder_to_reduce: Path, destiny_folder: Path):
    try:
        destiny_folder.mkdir(exist_ok=False)
    except FileExistsError:
        raise FileExistsError(f"La carpeta {destiny_folder.name} ya existe.")

    for file in folder_to_reduce.rglob("*.png"):
        image_arr: np.ndarray = MosaicGenerator.load_image(file)
        limg: Image.Image = Image.fromarray(image_arr)

        side = 200
        resize_arr = MosaicGenerator.resize_image(limg, (side, side))
        resize_img: Image.Image = Image.fromarray(resize_arr)

        resize_img.save(destiny_folder.joinpath(file.name))
        # MosaicGenerator.show_image_from_arr(image_arr, resize_arr)


if __name__ == "__main__":
    high_res_folder = Path(__file__).parent / "Images" / "Mosaic-Images-marina-cuadrada"
    new_folder = Path(__file__).parent / "Images" / "Mosaic-Images-marina-reduce"
    reduce_resolution(high_res_folder, new_folder)
