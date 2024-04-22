from typing import NamedTuple, Optional, overload
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
    @overload
    def __init(
        self,
        main_picture: Path | str,
        picture_suite: Path | str,
        export_path: Path | str,
        canvas_width: int,
        main_transparency: float = 0.67,
        mini_pic_res: Size = None,
    ) -> None: ...

    @overload
    def __init(
        self,
        main_picture: Path | str,
        picture_suite: Path | str,
        export_path: Path | str,
        canvas_height: int,
        main_transparency: float = 0.67,
        mini_pic_res: Size = None,
    ) -> None: ...

    def __init__(
        self,
        main_picture: Path | str,
        picture_suite: Path | str,
        export_path: Path | str,
        main_transparency: float = 0.67,
        canvas_res: Optional[Size] = None,
        mini_pic_res: Optional[Size] = None,
        canvas_width: Optional[int] = None,
        canvas_height: Optional[int] = None,
    ) -> None:
        self._main_picture: Path | str = main_picture
        self._picture_suite: Path | str = picture_suite
        self._export_path: Path | str = Path(export_path)
        self._main_transparency: float = main_transparency

        self.mini_pic_res: Size = mini_pic_res

        self._validate_paths()
        self._main_img: Image.Image = Image.open(main_picture)

        if canvas_res:
            self.__canvas_res(canvas_res)
        elif canvas_width:
            self.__init_width(canvas_width)
        elif canvas_height:
            self.__init_height(canvas_height)

        self._canvas: Image.Image = self.__create_canvas()

    def __init_width(self, width: int):
        wpx, hpx = self._main_img.size
        height = int(round((hpx / wpx) * width,0))
        self._canvas_res = Size(height=height, width=width)

    def __init_height(self, height):
        wpx, hpx = self._main_img.size
        width = int(round((wpx / hpx) * height,0))
        self._canvas_res = Size(height=height, width=width)

    def __canvas_res(self, canvas_res: int):
        self._canvas_res = canvas_res

    @property
    def canvas(self) -> Image.Image:
        return self._canvas

    @canvas.setter
    def canvas(self, value: Image.Image) -> None:
        self._canvas = value

    # region public methods
    def generate(self) -> None:
        img_list = self._create_image_list()

        self._paste_pics_in_canvas(img_list)
        self._add_base_image_to_canvas()
        return None

    def save(self) -> None:
        export_name = (
            f"{self._canvas_res.width}x{self._canvas_res.height}-"
            f"min_res {self.mini_pic_res.width}x{self.mini_pic_res.height}-"
            "Export.png"
        )

        path: Path = self._export_path / export_name
        self.canvas.save(path)
        return path

    # endregion

    # region Private methods
    def __create_canvas(self) -> Image.Image:
        return Image.new(
            "RGBA",
            (
                self.mini_pic_res.width * self._canvas_res.width,
                self.mini_pic_res.height * self._canvas_res.height,
            ),
        )

    def _create_image_list(self) -> list[np.ndarray]:
        images: list[np.ndarray] = []

        lista = list(self._picture_suite.rglob("*.png"))
        for file in lista:
            image_arr: np.ndarray = self.convert_image_path_to_array(file)

            if self.mini_pic_res is None:
                self.mini_pic_res = Size(*image_arr.shape[:-1])

            if image_arr.ndim == 3 and image_arr.shape[-1] == 4:
                limg = Image.fromarray(image_arr)
                resize_img = self._resize_image(limg, self.mini_pic_res)
                images.append(resize_img)
            else:
                pass
                # self.show_image_from_arr(image_arr)
        return images

    @staticmethod
    def convert_image_path_to_array(source: Path) -> np.ndarray:
        """Opens an image from specified source and returns a numpy array with image rgb data"""
        with Image.open(source) as im:
            im_arr = np.asarray(im)
        return im_arr

    @classmethod
    def _resize_image(cls, img: Image, shape: Size) -> np.ndarray:
        """Takes an image and resizes to a given size (width, height) as passed to the size parameter"""

        resz_img = ImageOps.fit(
            img,
            shape,
            Image.LANCZOS,
            centering=(0.5, 0.5),
        )
        resz_img = cls._apply_transparency(resz_img, transparency=1)
        return np.array(resz_img)

    def _paste_pics_in_canvas(self, img_list: np.ndarray) -> None:
        images_array: np.ndarray = np.asarray(img_list)
        image_values = np.apply_over_axes(np.mean, images_array, [1, 2]).reshape(len(img_list), 4)
        mos_template: np.ndarray = self._create_mos_template()
        tree = spatial.KDTree(image_values)

        for i in range(self._canvas_res.height):
            for j in range(self._canvas_res.width):
                template = mos_template[i, j]

                match = tree.query(template, k=40)
                pick = random.randint(0, 39)

                arr = img_list[match[1][pick]]
                x, y = j * self.mini_pic_res.width, i * self.mini_pic_res.height
                im = Image.fromarray(arr)
                self.canvas.paste(im, (x, y))
        return None

    def _create_mos_template(self):
        face_im_arr = self.convert_image_path_to_array(self._main_picture)

        alpha_channel = np.ones((face_im_arr.shape[0], face_im_arr.shape[1]), dtype=np.uint8) * 255
        face_im_arr_with_alpha = np.dstack((face_im_arr, alpha_channel))

        MAX_HEIGHT = face_im_arr_with_alpha.shape[0]
        MAX_WIDTH = face_im_arr_with_alpha.shape[1]

        return face_im_arr_with_alpha[
            :: (MAX_HEIGHT // self._canvas_res.height), :: (MAX_WIDTH // self._canvas_res.width)
        ]

    def _add_base_image_to_canvas(self) -> None:
        # Abrir ambas imágenes
        imagen_base = self.canvas.copy()
        main_img = Image.open(self._main_picture)

        # Ajustar tamaño de la imagen superpuesta si es necesario
        main_img = main_img.resize(imagen_base.size)

        # Agregar canal alfa a la imagen superpuesta
        main_img = main_img.convert("RGBA")

        # Ajustar la opacidad de la imagen superpuesta
        main_img_arr = np.asarray(main_img).copy()
        main_img_arr = main_img_arr.astype("float64")
        main_img_arr[:, :, 3] *= self._main_transparency

        main_img_arr = main_img_arr.astype("uint8")
        main_img = Image.fromarray(main_img_arr)

        # Combinar ambas imágenes
        self.canvas = Image.alpha_composite(imagen_base, main_img)
        return None

    @staticmethod
    def _show_image_from_arr(arr1: np.ndarray, arr2: Optional[np.ndarray] = None) -> None:
        _, axs = plt.subplots(1, 2, figsize=(10, 5))

        axs[0].imshow(Image.fromarray(arr1))
        axs[0].axis("off")
        axs[0].set_title("Imagen real")
        if arr2 is not None:
            axs[1].imshow(Image.fromarray(arr2))
            axs[1].axis("off")
            axs[1].set_title("Imagen recortada")

        plt.show()

    def _validate_paths(self) -> bool:
        if not self._main_picture.exists():
            raise FileNotFoundError(self._main_picture)

        if not self._picture_suite.exists():
            raise FileNotFoundError(self._picture_suite)

        if not self._export_path.exists():
            raise FileNotFoundError(self._export_path)
        return True

    @staticmethod
    def _apply_transparency(img: Image.Image, transparency: float) -> Image:
        """Applies transparency to the image"""
        # Convierte la imagen a modo RGBA si no lo está ya
        img = img.convert("RGBA")

        # Obtiene los datos de píxel de la imagen
        datos_pixeles = img.getdata()

        # Aplica transparencia a cada píxel
        nueva_lista_pixeles = []
        for pixel in datos_pixeles:
            r, g, b, a = pixel
            nuevo_valor_alfa = int(a * transparency)
            nuevo_pixel = (r, g, b, nuevo_valor_alfa)
            nueva_lista_pixeles.append(nuevo_pixel)

        # Actualiza los datos de píxel en la imagen
        img.putdata(nueva_lista_pixeles)

        return img

    # endregion
