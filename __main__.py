from pathlib import Path
from src import MosaicGenerator, Size


if __name__ == "__main__":
    downloads = Path.home()/"Downloads"
    main_pic = downloads/"facemo_project"/"main_image"/ "marina_mascarilla.png"
    pic_suite = downloads/"facemo_project"/"square_pictures"/"300pixels"

    mosaico = MosaicGenerator(
        main_picture=main_pic,
        picture_suite=pic_suite,
        canvas_width=1000,
        # target_res=Size(63,94),
        mini_pic_res=Size(1, 1),
    )
    mosaico.generate()
    mosaico.canvas.show()
    mosaico.save(downloads/"aniversario"/"fotos finales")

    print("Mosaico realizado con exito")
