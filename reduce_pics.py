import sys

from pathlib import Path
from PIL import Image
import numpy as np

from source.script import MosaicGenerator


def reduce_resolution(folder_to_reduce: Path, destiny_folder: Path, size: int = 200):
    try:
        destiny_folder.mkdir(exist_ok=False)
    except FileExistsError:
        raise FileExistsError(f"La carpeta {destiny_folder.name} ya existe.")

    for file in folder_to_reduce.rglob("*.png"):
        image_arr: np.ndarray = MosaicGenerator.load_image(file)
        limg: Image.Image = Image.fromarray(image_arr)
        
        resize_arr = MosaicGenerator.resize_image(limg, (size, size))
        resize_img: Image.Image = Image.fromarray(resize_arr)

        resize_img.save(destiny_folder.joinpath(file.name))
        # MosaicGenerator.show_image_from_arr(image_arr, resize_arr)


def validate_sys_argv(arg: str) -> tuple[str, int]:
    ...


if __name__ == "__main__":

    high_res_folder = Path(__file__).parent / "Images" / "Mosaic-Images-marina-cuadrada"

    default_folder_output:str = "Mosaic-Images-marina-reduce"
    default_new_folder = Path(__file__).parent / "Images"
    default_size = 200

    sys.argv.remove(sys.argv[0])

    n = len(sys.argv)
    if n == 1:
        var = sys.argv.pop()
        try:
            var = int(var)
            new_name = default_folder_output
            new_size = var
        except ValueError:
            new_name = var
            new_size = default_size

        # new_name, new_size = validate_sys_argv(var)

    elif n == 2:
        new_name = str(sys.argv[0])
        new_size = int(sys.argv[1])

    elif n > 2:
        raise ValueError("Se han pasado mas de 2 argumentos")
    else:
        new_name = default_folder_output
        new_size = default_size

    new_folder = default_new_folder / new_name

    reduce_resolution(high_res_folder, new_folder, size=new_size)
