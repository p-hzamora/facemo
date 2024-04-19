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
    def __init__(
        self,
        main_picture: Path | str,
        picture_suite: Path | str,
        target_res: Size,
        export_path: Path | str,
        mini_pic_res: Size = None,
    ) -> None:
        self._main_picture: Path | str = main_picture
        self._picture_suite: Path | str = picture_suite
        self._target_res: Size = target_res
        self._export_path: Path | str = Path(export_path)
        self.mini_pic_res: Size = mini_pic_res

    @staticmethod
    def show_image_from_arr(arr1: np.ndarray, arr2: Optional[np.ndarray] = None) -> None:
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
    def resize_image(cls, img: Image, shape: Size) -> np.ndarray:
        """Takes an image and resizes to a given size (width, height) as passed to the size parameter"""

        resz_img = ImageOps.fit(
            img,
            shape,
            Image.LANCZOS,
            centering=(0.5, 0.5),
        )
        return np.array(resz_img)

    def generate(self):
        img_list = self.create_image_list()
        canvas = Image.new(
            "RGB",
            (
                self.mini_pic_res.width * self._target_res.width,
                self.mini_pic_res.height * self._target_res.height,
            ),
        )

        self.paste_pics_in_canvas(img_list, canvas)

        self.save(canvas)

    def paste_pics_in_canvas(self, img_list: np.ndarray, canvas: Image.Image) -> None:
        images_array: np.ndarray = np.asarray(img_list)
        image_values = np.apply_over_axes(np.mean, images_array, [1, 2]).reshape(len(img_list), 3)
        mos_template: np.ndarray = self.create_mos_template()
        tree = spatial.KDTree(image_values)

        for i in range(self._target_res.height):
            for j in range(self._target_res.width):
                template = mos_template[i, j]

                match = tree.query(template, k=40)
                pick = random.randint(0, 39)

                arr = img_list[match[1][pick]]
                x, y = j * self.mini_pic_res.width, i * self.mini_pic_res.height
                im = Image.fromarray(arr)
                canvas.paste(im, (x, y))
        return None

    def create_mos_template(self):
        face_im_arr = self.load_image(self._main_picture)

        MAX_HEIGHT = face_im_arr.shape[0]
        MAX_WIDTH = face_im_arr.shape[1]

        return face_im_arr[:: (MAX_HEIGHT // self._target_res.height), :: (MAX_WIDTH // self._target_res.width)]

    def create_image_list(self) -> list[np.ndarray]:
        images: list[np.ndarray] = []

        lista = list(self._picture_suite.rglob("*.png"))
        for file in lista:
            image_arr: np.ndarray = self.load_image(file)

            if self.mini_pic_res is None:
                self.mini_pic_res = Size(*image_arr.shape[:-1])

            if image_arr.ndim == 3 and image_arr.shape[-1] == 3:
                limg = Image.fromarray(image_arr)
                resize_img = self.resize_image(limg, self.mini_pic_res)
                images.append(resize_img)
            else:
                pass
                # self.show_image_from_arr(image_arr)
        return images

    def save(self, canvas: Image.Image) -> None:
        export_name = (
            f"{self._target_res.height}x{self._target_res.height}-"
            f"min_res {self.mini_pic_res.width}x{self.mini_pic_res.height}-"
            "Export.jpg"
        )
        canvas.save(self._export_path / export_name)
        return None
