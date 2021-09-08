from al.train_test_func import *
from al.al_data_loader import QueryRepo
from al.al_utils import qbox_al_mainloop, qbox_al_scoring
from utils.datasets import ConvertRepo2Dataset
import alipy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='voc', help="['voc', 'kitti', 'kitti', 'covid']")
parser.add_argument('--folds', type=int, default=1)
parser.add_argument('--start-iter', type=int, default=0)
parser.add_argument('--end-iter', type=int, default=20)

parser.add_argument('--data-split-dir', type=str, default='/data/dataset/', help='dir of the data split files.')
parser.add_argument('--save-dir', type=str, default='/data/saved_model/', help='dir for saving model parameters and performances.')
parser.add_argument('--al-save-dir', type=str, default='/data/saved_al/', help='dir for saving the intermediate results of active learning.')
parser.add_argument('--save-name', type=str, default='', help='name of the model saving file')
parser.add_argument('--al-save-name', type=str, default='', help='name of the al saving file')

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
    # If you are using a new dataset, modify here accordingly
    raise ValueError("dataset must in ['voc', 'city', 'kitti', 'covid']")

setting_root = opt.data_split_dir
batch_size = 32
accumulate = 2
epochs = 50
nc = dataset_config['nc']
img_size = 416
cache_images = True
qbox_hyp = QBoxParam(opt.dataset)

with open(dataset_config['label_map'], 'rb') as f:
    src2tgt_labelmap = pickle.load(f)

for ifold, fold in enumerate(np.arange(start=0, stop=opt.folds)):
    if opt.folds > 1:
        assert opt.dataset == 'covid'
    init_seeds(seed=fold)

    # load data
    _, s_gt_ds, s_gt_dl = get_gt_dataloader(data=dataset_config['data'], data_item='source_train', img_size=img_size,
                                            batch_size=batch_size,
                                            rect=False, img_weights=False, cache_images=True, shuffle=False,
                                            augment=False, data_root=dataset_config['src_dir'], fold=fold)
    _, test_ds, test_dl = get_gt_dataloader(data=dataset_config['data'], data_item='valid', img_size=img_size,
                                            batch_size=batch_size,
                                            rect=False, img_weights=False, cache_images=True, shuffle=False,
                                            augment=False, data_root=dataset_config['tgt_dir'], fold=fold)

    # load target data
    # load init lab_unlab
    with open(os.path.join(setting_root, dataset_config['tgt'] + f'_tgt_ini{"_" + str(fold) if fold > 0 else ""}.txt'),
              'r') as f:
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
    with open(os.path.join(setting_root, dataset_config['src'] + f'_src_ini{"_" + str(fold) if fold > 0 else ""}.txt'),
              'r') as f:
        init_lab = f.read().splitlines(keepends=False)
    with open(
            os.path.join(setting_root, dataset_config['src'] + f'_src_unlab{"_" + str(fold) if fold > 0 else ""}.txt'),
            'r') as f:
        init_unlab = f.read().splitlines(keepends=False)

    # start pipeline
    unlab_len = len(init_unlab)
    l_set = alipy.index.IndexCollection(init_lab)
    ul_set = alipy.index.IndexCollection(init_unlab)
    queried_repo = QueryRepo()
    budget_arr = np.array([dataset_config['budget']] * 100) # amount to fully annotating 100 randomly selected examples. 5 for COVID dataset

    # load ini model
    model, optimizer, _ = load_voc_model(
        pt_path=os.path.join(opt.save_dir, f'small_{dataset_config["tgt"]}{"_" + str(fold) if fold > 0 else ""}.pt'),
        cfg=f'cfg/yolov3-spp-{dataset_config["tgt"]}.cfg',
        parallel=True, parallel_port=6666, init_group=True if ifold == 0 else False)

    for i in np.arange(opt.start_iter, opt.end_iter):
        model.eval()
        func_ul_set = ul_set
        with torch.no_grad():
            scores = qbox_al_scoring(unlab_arr=func_ul_set,
                                     model=model, s_gt_ds=s_gt_ds,
                                     cocoid2vocid=src2tgt_labelmap,
                                     queried_repo=queried_repo,
                                     pos_ins_weight=qbox_hyp.pos_ins_weight,
                                     da_tradeoff=qbox_hyp.da_tradeoff,
                                     min_area=dataset_config['min_area'])

        queried_repo, total_cost, num_stat = qbox_al_mainloop(scoring_arr=scores, src_gt_ds=s_gt_ds,
                                                              budget=budget_arr[i],
                                                              queried_repo=queried_repo, iteration=str(i),
                                                              src2tgt_label_map=src2tgt_labelmap,
                                                              save_suffix=opt.al_save_name, fold=fold,
                                                              save_root=opt.al_save_dir)

        updated_lab_ds = ConvertRepo2Dataset(query_repo=queried_repo, img_size=img_size, batch_size=batch_size,
                                             augment=True, hyp=hyp, rect=False, image_weights=False, cache_images=True)
        ini_src_ds = LoadImagesAndLabelsByImgFiles(
            img_files=init_lab if len(queried_repo.fs_database) == 0 else init_lab + queried_repo.fs_database + queried_repo.fs_database,
            img_size=img_size,
            batch_size=batch_size,
            augment=True,
            hyp=hyp,
            rect=False,
            image_weights=False,
            cache_images=cache_images
        )

        # Dataloader
        q_dataloader = torch.utils.data.DataLoader(updated_lab_ds,
                                                   batch_size=batch_size,
                                                   num_workers=0,
                                                   shuffle=False,  # Shuffle=True unless rectangular training is used
                                                   pin_memory=True,
                                                   collate_fn=updated_lab_ds.collate_fn,
                                                   drop_last=False)

        # Dataloader
        ini_dataloader = torch.utils.data.DataLoader(ini_src_ds,
                                                     batch_size=batch_size,
                                                     num_workers=0,
                                                     shuffle=False,  # Shuffle=True unless rectangular training is used
                                                     pin_memory=True,
                                                     collate_fn=ini_src_ds.collate_fn,
                                                     drop_last=False)

        model, best_ap = train_mix_partial(model=model, optimizer=optimizer, dataloader=ini_dataloader,
                                           queried_dataloader=q_dataloader, tgt_dataloader=t_gt_dl,
                                           start_epoch=0, epochs=epochs, nc=nc, batch_size=batch_size,
                                           src2tgt_label_map=src2tgt_labelmap,
                                           test_dl=test_dl, ins_gain=5,
                                           saved_map_dir=os.path.join(opt.save_dir, f"qbox_{i}_saved_map{('_' + opt.save_name) if opt.save_name else ''}{('_' + str(fold)) if fold > 0 else ''}.txt"),
                                           partial_loss_hyp=qbox_hyp)
