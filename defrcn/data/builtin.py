import os
from .meta_coco import register_meta_coco
from .meta_main import register_meta_main
from .builtin_meta import _get_builtin_metadata
from detectron2.data import DatasetCatalog, MetadataCatalog


# -------- COCO -------- #
def register_all_coco(root="datasets"):

    METASPLITS = [
        ("coco14_trainval_all", "coco/trainval2014", "cocosplit/datasplit/trainvalno5k.json"),
        ("coco14_trainval_base", "coco/trainval2014", "cocosplit/datasplit/trainvalno5k.json"),
        ("coco14_test_all", "coco/val2014", "cocosplit/datasplit/5k.json"),
        ("coco14_test_base", "coco/val2014", "cocosplit/datasplit/5k.json"),
#         ("coco14_test_novel", "coco/val2014", "cocosplit/datasplit/5k.json"),
    ]
    for prefix in ["all", "novel"]:
        for shot in [1, 2, 3, 5, 10, 30]:
            for seed in range(10):
                name = "coco14_trainval_{}_{}shot_seed{}".format(prefix, shot, seed)
                METASPLITS.append((name, "coco/trainval2014", ""))

    for name, imgdir, annofile in METASPLITS:
        register_meta_coco(
            name,
            _get_builtin_metadata("coco_fewshot"),
            os.path.join(root, imgdir),
            os.path.join(root, annofile),
        )

def register_all_main(root="datasets"):

    METASPLITS = [
        ("main_2022_train_10shot", "main", "train", "base1", 1),
        ("main_2022_test", "main", "test", "base1", 2),
        ("main_2022_test2", "main", "test", "base2", 1),
        ("main_2022_test3", "main", "test", "novel1", 3),
        ("main_2022_test4", "main", "test_short", "base1", 2),
        ("main_2022_test5", "main", "test_short2", "base1", 2),
    ]


    for name, dirname, split, keepclasses, sid in METASPLITS:
        register_meta_main(
            name,
            _get_builtin_metadata("main_fewshot"),
            os.path.join(root, dirname),
            split,
            keepclasses,
            sid,
        )
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"

register_all_coco()
register_all_main()