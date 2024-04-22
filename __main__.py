from pathlib import Path
from source.script import MosaicGenerator, Size


if __name__ == "__main__":
    main_pic = Path(__file__).parent / "Images" / "segunda opcion.jpg"
    pic_suite = Path(__file__).parent.parent / "720pixels"
    export = Path(__file__).parent / "Images" / "Exports"

    mosaico = MosaicGenerator(
        main_picture=main_pic,
        picture_suite=pic_suite,
        canvas_width=56,
        # target_res=Size(63,94),
        export_path=export,
        mini_pic_res=Size(350, 350),
    )
    mosaico.generate()
    mosaico.canvas.show()
    mosaico.save()

    print("Mosaico realizado con exito")
