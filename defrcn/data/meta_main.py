import os
import numpy as np
import xml.etree.ElementTree as ET
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog


__all__ = ["register_meta_main"]


def load_filtered_voc_instances(
    name: str, dirname: str, split: str, classnames: str
):
    """
    Load Pascal VOC detection annotations to Detectron2 format.
    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    is_shots = "shot" in name
    if is_shots:
        fileids = {}
        split_dir = os.path.join("datasets", "main", "shot")
        splittted = name.split("_")
        shot = name.split("_")[-1].split("shot")[0]
        # seed = int(name.split("_seed")[-1])
        # split_dir = os.path.join(split_dir, "seed{}".format(seed))
        for cls in classnames:
            with PathManager.open(
                os.path.join(
                    split_dir, "{}_{}shot.txt".format(cls, shot)
                )
            ) as f:
                fileids_ = np.loadtxt(f, dtype=np.str).tolist()
                if isinstance(fileids_, str):
                    fileids_ = [fileids_]
                fileids_ = [
                    fid.split("/")[-1].split(".jpg")[0] for fid in fileids_
                ]
                fileids[cls] = fileids_
    else:
        fn1 = os.path.join(dirname, "ImageSets", "Main", split + ".txt")
        print("path is {}".format(fn1))
        with PathManager.open(
            os.path.join(dirname, "ImageSets", "Main", split + ".txt")
        ) as f:
            fileids = np.loadtxt(f, dtype=np.str)

    dicts = []
    if is_shots:
        for cls, fileids_ in fileids.items():
            dicts_ = []
            for fileid in fileids_:
                dirname = os.path.join("datasets", "main")
                anno_file = os.path.join(
                    dirname, "Annotations", fileid + ".xml"
                )
                jpeg_file = os.path.join(
                    dirname, "JPEGImages", fileid + ".jpg"
                )

                tree = ET.parse(anno_file)

                for obj in tree.findall("object"):
                    r = {
                        "file_name": jpeg_file,
                        "image_id": fileid,
                        "height": int(tree.findall("./size/height")[0].text),
                        "width": int(tree.findall("./size/width")[0].text),
                    }
                    cls_ = obj.find("name").text
                    if cls != cls_:
                        continue
                    bbox = obj.find("bndbox")
                    bbox = [
                        float(bbox.find(x).text)
                        for x in ["xmin", "ymin", "xmax", "ymax"]
                    ]
                    bbox[0] -= 1.0
                    bbox[1] -= 1.0

                    instances = [
                        {
                            "category_id": classnames.index(cls),
                            "bbox": bbox,
                            "bbox_mode": BoxMode.XYXY_ABS,
                        }
                    ]
                    r["annotations"] = instances
                    dicts_.append(r)
            if len(dicts_) > int(shot):
                dicts_ = np.random.choice(dicts_, int(shot), replace=False)
            dicts.extend(dicts_)
    else:
        for fileid in fileids:
            anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
            jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

            tree = ET.parse(anno_file)

            r = {
                "file_name": jpeg_file,
                "image_id": fileid,
                "height": int(tree.findall("./size/height")[0].text),
                "width": int(tree.findall("./size/width")[0].text),
            }
            instances = []

            for obj in tree.findall("object"):
                cls = obj.find("name").text
                if not (cls in classnames):
                    continue
                bbox = obj.find("bndbox")
                bbox = [
                    float(bbox.find(x).text)
                    for x in ["xmin", "ymin", "xmax", "ymax"]
                ]
                bbox[0] -= 1.0
                bbox[1] -= 1.0

                instances.append(
                    {
                        "category_id": classnames.index(cls),
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                    }
                )
            r["annotations"] = instances
            dicts.append(r)
    return dicts


def register_meta_main(
    name, metadata, dirname, split, keepclasses, sid
):
    if keepclasses.startswith("base1"):
        thing_classes = metadata["thing_classes"][sid]
    elif keepclasses.startswith("base2"):
        thing_classes = metadata["base_classes"][sid]
    elif keepclasses.startswith("novel1"):
        thing_classes = metadata["novel_classes"][sid]

    DatasetCatalog.register(
        name,
        lambda: load_filtered_voc_instances(
            name, dirname, split, thing_classes
        ),
    )

    MetadataCatalog.get(name).set(
        thing_classes=thing_classes,
        dirname=dirname,
        split=split,
        year=2007,
        base_classes=metadata["base_classes"][sid],
        novel_classes=metadata["novel_classes"][sid],
    )
