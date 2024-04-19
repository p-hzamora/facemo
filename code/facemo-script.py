from typing import NamedTuple, Optional
from PIL import Image, ImageOps
import numpy as np
from matplotlib import pyplot as plt
from scipy import spatial
import random
from pathlib import Path


class Size(NamedTuple):
    width: int
    height: int


class MosaicGenerator:
    mini_pic_res = Size(40, 40)

    def __init__(
        self,
        main_picture: Path | str,
        picture_suite: Path | str,
        target_res: Size,
        export_path: Path | str,
    ) -> None:
        self._main_picture: Path | str = main_picture
        self._picture_suite: Path | str = picture_suite
        self._target_res: Size = target_res
        self._export_path: Path | str = Path(export_path)

    @staticmethod
    def show_image_from_arr(
        arr1: np.ndarray, arr2: Optional[np.ndarray] = None
    ) -> None:
        _, axs = plt.subplots(1, 2, figsize=(10, 5))

        axs[0].imshow(Image.fromarray(arr1))
        axs[0].axis("off")
        axs[0].set_title("Imagen real")
        if arr2 is not None:
            axs[1].imshow(Image.fromarray(arr2))
            axs[1].axis("off")
            axs[1].set_title("Imagen recortada")

        plt.show()

    @staticmethod
    def load_image(source: Path) -> np.ndarray:
        """Opens an image from specified source and returns a numpy array with image rgb data"""
        with Image.open(source) as im:
            im_arr = np.asarray(im)
        return im_arr

    @classmethod
    def resize_image(cls, img: Image, shape: Size = None) -> np.ndarray:
        """Takes an image and resizes to a given size (width, height) as passed to the size parameter"""

        shape = shape if shape is not None else cls.mini_pic_res
        resz_img = ImageOps.fit(
            img,
            shape,
            Image.LANCZOS,
            centering=(0.5, 0.5),
        )
        return np.array(resz_img)

    def generate(self):
        face_im_arr = self.load_main_picture()

        height = face_im_arr.shape[0]
        width = face_im_arr.shape[1]

        mos_template: np.ndarray = face_im_arr[
            :: (height // self._target_res.height), :: (width // self._target_res.width)
        ]
        # self.show_image_from_arr(mos_template)
        # self.show_image_from_arr(face_im_arr, mos_template)

        images = self.crop_mosaic_image()

        # self.show_image_from_arr(images[4])

        images_array: np.ndarray = np.asarray(images)

        image_values = np.apply_over_axes(np.mean, images_array, [1, 2]).reshape(
            len(images), 3
        )

        tree = spatial.KDTree(image_values)

        image_idx = np.zeros(self._target_res, dtype=np.uint32)

        for i in range(self._target_res.height):
            for j in range(self._target_res.width):
                template = mos_template[i, j]

                match = tree.query(template, k=40)
                pick = random.randint(0, 39)
                image_idx[j, i] = match[1][pick]

        canvas = Image.new(
            "RGB",
            (
                self.mini_pic_res.height * self._target_res.height,
                self.mini_pic_res.width * self._target_res.width,
            ),
        )

        for i in range(self._target_res.height):
            for j in range(self._target_res.width):
                arr = images[image_idx[j, i]]
                x, y = j * self.mini_pic_res.width, i * self.mini_pic_res.height
                im = Image.fromarray(arr)
                canvas.paste(im, (x, y))

        canvas.save(self._export_path)

    def crop_mosaic_image(self) -> list[np.ndarray]:
        images: list[np.ndarray] = []

        lista = list(self._picture_suite.rglob("*.png"))
        for file in lista:
            image_arr: np.ndarray = self.load_image(file)

            if image_arr.ndim == 3:
                limg = Image.fromarray(image_arr)
                resize_img = self.resize_image(limg)
                images.append(resize_img)
            else:
                self.show_image_from_arr(image_arr)
        return images

    def load_main_picture(self) -> np.ndarray:
        face_im_arr = self.load_image(self._main_picture)
        return face_im_arr


# def reduce_resolution():
#     cuadradas_high_res = Path(__file__).parent.parent/"Images"/"Mosaic-Images-marina-cuadrada"
#     for file in cuadradas_high_res.rglob("*.png"):
#         image_arr: np.ndarray = MosaicGenerator.load_image(file)
#         limg: Image.Image = Image.fromarray(image_arr)

#         resize_arr = MosaicGenerator.resize_image(limg, (side, side))
#         resize_img: Image.Image = Image.fromarray(resize_arr)

#         new_file =
#         resize_img.save(new_file.with_suffix(".png"))
#         # MosaicGenerator.show_image_from_arr(image_arr, resize_arr)


if __name__ == "__main__":
    main_pic = Path(__file__).parent.parent / "Images" / "cuadrada_mano.png"
    pic_suite = Path(__file__).parent.parent / "Images" / "Mosaic-Images-real"
    export = (
        Path(__file__).parent.parent / "Images" / "Exports" / "exportacion_marina.jpg"
    )

    MosaicGenerator(
        main_picture=main_pic,
        picture_suite=pic_suite,
        target_res=Size(150, 100),
        export_path=export,
    ).generate()

    print("Mosaico realizado con exito")
