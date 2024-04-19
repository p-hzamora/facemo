from pathlib import Path
from source.script import MosaicGenerator, Size

if __name__ == "__main__":

    main_pic = Path(__file__).parent / "Images" / "cuadrada_mano.png"
    pic_suite = Path(__file__).parent / "Images" / "Mosaic-Images-real"
    export = Path(__file__).parent / "Images" / "Exports" / "exportacion_marina.jpg"

    MosaicGenerator(
        main_picture=main_pic,
        picture_suite=pic_suite,
        target_res=Size(150, 100),
        export_path=export,
    ).generate()

    print("Mosaico realizado con exito")
