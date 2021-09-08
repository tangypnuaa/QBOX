# path to the darknet53.conv.74 weight file
weights_path='~/todal/darknet/darknet53.conv.74'

VOC_COCO = {
    'src': 'coco',  # name of the source dataset. Should be accord with the split file, cfg file, .data file, .names file, etc.
    'tgt': 'voc',   # name of the target dataset. Should be accord with the split file, cfg file, .data file, .names file, etc.
    'pkl': '/data/saved_model/init_da_yolo_voc.pkl',    # initialization model path (to eliminate the randomness)
    'src_dir': '/data/coco/images/train2014/',
    'tgt_dir': '/data/voc/voc_pure_data/images/',
    'data': 'data/voc_coco.data',       # path to the .data file
    'label_map': 'data/cocoid2vocid.pkl',   # label map pickle file
    'cfg': 'cfg/yolov3-spp-voc.cfg',    # path to the model cfg file
    'nc': 20,   # number of target classes
    'budget': 13369,   # budget for one query, amount to fully annotating 100 randomly selected examples.
    'min_area':112       # the 10th percentile of the box area in the initially labeled set
}

KITTI_CITY = {
    'src': 'city',
    'tgt': 'kitti',
    'pkl': '/data/saved_model/init_da_yolo_kitti.pkl',
    'src_dir': '/data/cityscape/coco/images/',
    'tgt_dir': '/data/download/kitti/tgt_images/',
    'data': 'data/kitti_city.data',
    'label_map': 'data/city2kitti.pkl',
    'cfg': 'cfg/yolov3-spp-kitti.cfg',
    'nc': 5,
    'budget': 44564,
    'min_area': 37
}

CITY_VOC = {
    'src': 'voc',
    'tgt': 'city',
    'pkl': '/data/saved_model/init_da_yolo_city.pkl',
    'src_dir': '/data/voc/voc_pure_data/images/',
    'tgt_dir': '/data/cityscape/coco/images_tgt/',
    'data': 'data/city_voc.data',
    'label_map': 'data/voc2city.pkl',
    'cfg': 'cfg/yolov3-spp-city.cfg',
    'nc': 6,
    'budget': 9804,
    'min_area':755
}

COVID_PNEU = {
    'src': 'pneu',
    'tgt': 'covid',
    'pkl': '/data/saved_model/covid_xls_pretrain.pkl',
    'src_dir': None,
    'tgt_dir': None,
    'data': 'data/covid.data',
    'label_map': 'data/covid_label.pkl',
    'cfg': 'cfg/yolov3-spp-covid.cfg',
    'nc': 2,
    'budget': (25.5*2 + 7.8) * 5,
    'min_area':2500
}

