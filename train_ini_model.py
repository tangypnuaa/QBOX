from al.train_test_func import *
from al.al_utils import al_mainloop, al_scoring
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='voc', help="['voc', 'kitti', 'sim', 'covid']")
parser.add_argument('--ins-gain', type=float, default=5.0)
parser.add_argument('--save-name', type=str, default='', help='name of the saved file')
parser.add_argument('--folds', type=int, default=1)
parser.add_argument('--start-fold', type=int, default=0)
parser.add_argument('--data-split-dir', type=str, default='/data/dataset/', help='dir of the data split files.')
parser.add_argument('--save-dir', type=str, default='/data/saved_model/', help='dir for saving model parameters and performances.')

opt = parser.parse_args()

if opt.dataset == 'voc':
    dataset_config = VOC_COCO
elif opt.dataset == 'kitti':
    dataset_config = KITTI_CITY
elif opt.dataset == 'city':
    dataset_config = CITY_VOC
elif opt.dataset == 'covid':
    dataset_config = COVID_PNEU
else:
    raise ValueError("dataset must in ['voc', 'sim', 'kitti', 'covid']")

setting_root = opt.data_split_dir
batch_size = 32
accumulate = 2
epochs = 50
img_size = 416
nc = dataset_config['nc']

with open(dataset_config['label_map'], 'rb') as f:
    src2tgt_labelmap = pickle.load(f)

for ifold, fold in enumerate(np.arange(start=opt.start_fold, stop=opt.folds)):
    if opt.folds > 1:
        assert opt.dataset == 'covid'
    init_seeds(seed=fold)

    # init model
    model, optimizer = init_model(pkl_path=dataset_config['pkl'],
                                  cfg=dataset_config['cfg'],
                                  weights_path='~/todal/darknet/darknet53.conv.74',
                                  init_group=False if (fold > opt.start_fold) else True)

    # load data
    _, s_gt_ds, s_gt_dl = get_gt_dataloader(data=dataset_config['data'], data_item='source_train', img_size=img_size,
                                            batch_size=batch_size,
                                            rect=False, img_weights=False, cache_images=True, shuffle=False,
                                            augment=False, data_root=dataset_config['src_dir'], fold=fold)
    _, test_ds, test_dl = get_gt_dataloader(data=dataset_config['data'], data_item='valid', img_size=img_size, batch_size=batch_size,
                                            rect=False, img_weights=False, cache_images=True, shuffle=False,
                                            augment=False, data_root=dataset_config['tgt_dir'], fold=fold)

    with open(os.path.join(setting_root, dataset_config['tgt']+f'_tgt_ini{"_"+str(fold) if fold > 0 else ""}.txt'), 'r') as f:
        init_lab = f.read().splitlines(keepends=False)


    t_gt_ds = LoadImagesAndLabelsByImgFiles(
        img_files=init_lab,
        img_size=img_size,
        batch_size=batch_size,
        augment=True,
        hyp=hyp,  # augmentation hyperparameters
        rect=False,  # rectangular training
        image_weights=False,
        cache_images=True
    )

    # Dataloader
    t_gt_dl = torch.utils.data.DataLoader(t_gt_ds,
                                         batch_size=batch_size,
                                         num_workers=4,
                                         shuffle=False,  # Shuffle=True unless rectangular training is used
                                         pin_memory=True,
                                         collate_fn=t_gt_ds.collate_fn)

    # load init lab_unlab
    with open(os.path.join(setting_root, dataset_config['src']+f'_src_ini{"_"+str(fold) if fold > 0 else ""}.txt'), 'r') as f:
        init_lab = f.read().splitlines(keepends=False)
    with open(os.path.join(setting_root, dataset_config['src']+f'_src_unlab{"_"+str(fold) if fold > 0 else ""}.txt'), 'r') as f:
        init_unlab = f.read().splitlines(keepends=False)

    # calc initial performance point
    ini_dataset = LoadImagesAndLabelsByImgFiles(
        img_files=init_lab,
        img_size=img_size,
        batch_size=batch_size,
        augment=True,
        hyp=hyp,  # augmentation hyperparameters
        rect=False,  # rectangular training
        image_weights=False,
        cache_images=True
    )

    # Dataloader
    ini_dataloader = torch.utils.data.DataLoader(ini_dataset,
                                                 batch_size=batch_size,
                                                 num_workers=0,
                                                 shuffle=False,  # Shuffle=True unless rectangular training is used
                                                 pin_memory=True,
                                                 collate_fn=ini_dataset.collate_fn)

    model, best_ap = train_mix(model=model, optimizer=optimizer, dataloader=ini_dataloader, tgt_dataloader=t_gt_dl,
                               start_epoch=0, epochs=epochs, nc=nc, batch_size=batch_size,
                               src2tgt_label_map=src2tgt_labelmap, save_epoch=[], notest=False, test_dl=test_dl, save_pt=True,
                               ins_gain=opt.ins_gain, best_save_name=os.path.join(opt.save_dir ,f'small_{dataset_config["tgt"]}{"_"+str(fold) if fold > 0 else ""}.pt'),
                               save_prefix='saved_ckpt_al_',
                               saved_map_dir=os.path.join(opt.save_dir ,f'small_{dataset_config["tgt"]}{opt.save_name}{"_"+str(fold) if fold > 0 else ""}.txt')


