# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------
# v11:original input(d, W, H), (heatmap v1) after c23
import math
import os
import random
import time

# import apex
import mathutils
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import visibility

from sacred import Experiment, SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds

from camera_model_mapping import CameraModel
from Dataset_kitti_mapping import DatasetVisibilityKittiSingle
from losses import DistancePoints3D, GeometricLoss, L1Loss, ProposedLoss
from models.CMRNet.CMRNet_single_save import CMRNet
# from models.CMRNet.CMRNet_h_RT_savep import CMRNet
from quaternion_distances import quaternion_distance
from utils import merge_inputs, overlay_imgs, rotate_back


datasetType = 0 # 0 --- kitti 1 --- argo

# from prefetch_generator import BackgroundGenerator
#
# class DataLoaderX(torch.utils.data.DataLoader):
#     def __iter__(self):
#         return BackgroundGenerator(super().__iter__())

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

SETTINGS.DISCOVER_DEPENDENCIES = "none"
SETTINGS.DISCOVER_SOURCES = "none"
ex = Experiment("CMRNet")
ex.captured_out_filter = apply_backspaces_and_linefeeds


# noinspection PyUnusedLocal
@ex.config
def config():
    savemodel = './checkpoints/'
    dataset = 'kitti'
    data_folder = './KITTI/sequences'
    use_reflectance = False
    test_sequence = 0
    occlusion_kernel = 5  # 3
    occlusion_threshold = 3  # 3.9999
    epochs = 300
    BASE_LEARNING_RATE = 1e-4  # 3e-4
    loss = 'simple'
    max_t = 2.
    max_r = 10.
    batch_size = 24  # 24
    num_worker = 3 #5
    network = 'PWC_f1'
    optimizer = 'adam'
    resume = None
    weights = None
    rescale_rot = 10
    rescale_transl = 1
    precision = "O0"
    norm = 'bn'
    dropout = 0.0
    max_depth = 100.
    maps_folder = 'local_maps_0.1'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCH = 1
def _init_fn(worker_id, seed):
    seed = seed + worker_id + EPOCH*100
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def flow_loss_fn(flow, duv, flag):
    # B, C, H, W = flow.size()
    flow = flag * flow
    fn = torch.nn.SmoothL1Loss(reduction='none')
    sum = torch.sum(flag, (0,1,2,3), keepdim=False)
    loss = fn(flow, duv).sum((0,1,2,3)) / sum
    return loss

# CCN training
@ex.capture
def train(model, optimizer, pcl, info, u, v, rgb_img, refl_img, flagt, depth, target_transl, target_rot, loss_fn, point_clouds, loss):
    model.train()

    optimizer.zero_grad()

    # Run model
    transl_err, rot_err, t0, r0 = model(rgb_img, refl_img, depth, flagt, u, v, pcl, info)

    # loss_q = torch.mean(torch.sqrt(torch.sum((target_rot-rot_err)*(target_rot-rot_err), dim=-1, keepdim=True)+1e-10))
    # loss_t = torch.mean(torch.sqrt((target_transl-transl_err)*(target_transl-transl_err)+1e-10))
    # total_loss = loss_t * torch.exp(-w_x) + w_x + loss_q * torch.exp(-w_q) + w_q
    if loss != 'points_distance':
        total_loss1 = loss_fn(target_transl, target_rot, transl_err, rot_err)
        total_loss2 = loss_fn(target_transl, target_rot, t0, r0)
    else:
        total_loss1 = loss_fn(point_clouds, target_transl, target_rot, transl_err, rot_err)
        total_loss2 = loss_fn(target_transl, target_rot, t0, r0)
    # with apex.amp.scale_loss(total_loss, optimizer) as scaled_loss:
    #     scaled_loss.backward()
    total_loss = 0.6*total_loss1 + 0.4*total_loss2
    total_loss.backward()
    optimizer.step()

    return total_loss.item()


# CNN test
@ex.capture
def test(model, pcl, info, u, v, rgb_img, refl_img, flagt, depth, target_transl, target_rot, loss_fn, camera_model, point_clouds, loss):
    model.eval()

    # Run model
    with torch.no_grad():
        transl_err, rot_err, t0, r0 = model(rgb_img, refl_img, depth, flagt, u, v, pcl, info)



    if loss != 'points_distance':
        total_loss = loss_fn(target_transl, target_rot, transl_err, rot_err)
    else:
        total_loss = loss_fn(point_clouds, target_transl, target_rot, transl_err, rot_err)

    total_trasl_error = torch.tensor(0.0)
    total_trasl_error = total_trasl_error.to(target_rot.device)
    total_rot_error = quaternion_distance(target_rot, rot_err, target_rot.device)
    total_rot_error = total_rot_error * 180. / math.pi
    for j in range(rgb_img.shape[0]):
        total_trasl_error += torch.norm(target_transl[j] - transl_err[j]) * 100.

    return total_loss.item(), total_trasl_error.item(), total_rot_error.sum().item()



@ex.automain
def main(_config, _run, seed):
    global EPOCH
    print(_config['loss'])

    if _config['test_sequence'] is None:
        raise TypeError('test_sequences cannot be None')
    else:
        if datasetType == 0:
            _config['test_sequence'] = f"{_config['test_sequence']:02d}"
        elif datasetType == 1:
            _config['test_sequence'] = _config['test_sequence']
        print("Test Sequence: ", _config['test_sequence'])
        dataset_class = DatasetVisibilityKittiSingle
    occlusion_threshold = _config['occlusion_threshold']
    if datasetType == 0:
        img_shape = (384, 1280)
    elif datasetType == 1:
        img_shape = (640, 960)
        
    _config["savemodel"] = os.path.join(_config["savemodel"], _config['dataset'])

    maps_folder = 'local_maps'
    if _config['maps_folder'] is not None:
        maps_folder = _config['maps_folder']
    dataset = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                            split='train', use_reflectance=_config['use_reflectance'], maps_folder=maps_folder,
                            test_sequence=_config['test_sequence'])
    dataset_val = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                split='test', use_reflectance=_config['use_reflectance'], maps_folder=maps_folder,
                                test_sequence=_config['test_sequence'])
    _config["savemodel"] = os.path.join(_config["savemodel"], _config['test_sequence'])
    if not os.path.exists(_config["savemodel"]):
        os.makedirs(_config["savemodel"])

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    def init_fn(x): return _init_fn(x, seed)

    dataset_size = len(dataset)

    # Training and test set creation
    num_worker = _config['num_worker']
    batch_size = _config['batch_size']
    TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset,
                                                 shuffle=False,
                                                 batch_size=batch_size,
                                                 num_workers=num_worker,
                                                 worker_init_fn=init_fn,
                                                 collate_fn=merge_inputs,
                                                 drop_last=False,
                                                 pin_memory=True)

    TestImgLoader = torch.utils.data.DataLoader(dataset=dataset_val,
                                                shuffle=False,
                                                batch_size=batch_size,
                                                num_workers=num_worker,
                                                worker_init_fn=init_fn,
                                                collate_fn=merge_inputs,
                                                drop_last=False,
                                                pin_memory=True)

    print(len(TrainImgLoader))
    print(len(TestImgLoader))

    if _config['loss'] == 'simple':
        loss_fn = ProposedLoss(_config['rescale_transl'], _config['rescale_rot'])
    elif _config['loss'] == 'geometric':
        loss_fn = GeometricLoss()
        loss_fn = loss_fn.to(device)
    elif _config['loss'] == 'points_distance':
        loss_fn = DistancePoints3D()
    elif _config['loss'] == 'L1':
        loss_fn = L1Loss(_config['rescale_transl'], _config['rescale_rot'])
    else:
        raise ValueError("Unknown Loss Function")

    #runs = datetime.now().strftime('%b%d_%H-%M-%S') + "/"
    #train_writer = SummaryWriter('./logs/' + runs)
    #ex.info["tensorflow"] = {}
    #ex.info["tensorflow"]["logdirs"] = ['./logs/' + runs]

    if _config['network'].startswith('PWC'):
        feat = 1 # 1 # 2 # 3
        md = 4
        split = _config['network'].split('_')
        for item in split[1:]:
            if item.startswith('f'):
                feat = int(item[-1])
            elif item.startswith('md'):
                md = int(item[2:])
        assert 0 < feat < 7, "Feature Number from PWC have to be between 1 and 6"
        assert 0 < md, "md must be positive"
        feat = 2
        model = CMRNet(img_shape, use_feat_from=feat, md=md,
                       use_reflectance=_config['use_reflectance'], dropout=_config['dropout'])
    else:
        raise TypeError("Network unknown")
    if _config['weights'] is not None:
        print(f"Loading weights from {_config['weights']}")
        checkpoint = torch.load(_config['weights'], map_location='cpu')
        saved_state_dict = checkpoint['state_dict']
        model.load_state_dict(saved_state_dict)
    model = model.to(device)

    print(dataset_size)
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    if _config['loss'] == 'geometric':
        parameters += list(loss_fn.parameters())
    if _config['optimizer'] == 'adam':
        optimizer = optim.Adam(parameters, lr=_config['BASE_LEARNING_RATE'], weight_decay=5e-6)
        # Probably this scheduler is not used
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 70], gamma=0.5)
    else:
        optimizer = optim.SGD(parameters, lr=_config['BASE_LEARNING_RATE'], momentum=0.9,
                              weight_decay=5e-6, nesterov=True)

    starting_epoch = 0
    if _config['weights'] is not None and _config['resume']:
        checkpoint = torch.load(_config['weights'], map_location='cpu')
        opt_state_dict = checkpoint['optimizer']
        optimizer.load_state_dict(opt_state_dict)
        starting_epoch = checkpoint['epoch']

    # Allow mixed-precision if needed
    # model, optimizer = apex.amp.initialize(model, optimizer, opt_level=_config["precision"])

    start_full_time = time.time()
    BEST_VAL_LOSS = 10000.
    old_save_filename = None

    total_iter = 0
    for epoch in range(starting_epoch, _config['epochs'] + 1):
        EPOCH = epoch
        # print('This is %d-th epoch' % epoch)
        epoch_start_time = time.time()
        total_train_loss = 0
        local_loss = 0.
        if _config['optimizer'] != 'adam':
            _run.log_scalar("LR", _config['BASE_LEARNING_RATE'] *
                            math.exp((1 - epoch) * 4e-2), epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = _config['BASE_LEARNING_RATE'] * \
                                    math.exp((1 - epoch) * 4e-2)
        else:
            #scheduler.step(epoch%100)
            _run.log_scalar("LR", scheduler.get_lr()[0])


        ## Training ##
        time_for_50ep = time.time()

        get_sample_time = 0
        # rotate_time = 0
        # project_time = 0
        # occlusion_time = 0
        train_time = 0
        start_time = time.time()
        for batch_idx, sample in enumerate(TrainImgLoader):
            # break
            # print(f'batch {batch_idx+1}/{len(TrainImgLoader)}', end='\r')

            lidar_input = []
            rgb_input = []
            duv_input = []
            depth_input = []
            flagt_input = []
            flag_input = []
            ltolu_input = []
            ltolv_input = []
            pcl_input = []
            info_input = []
            # pos_uv_input = []
            # pos_xy_input = []
            # print(time.time() - start_time)

            sample['tr_error'] = sample['tr_error'].cuda()
            sample['rot_error'] = sample['rot_error'].cuda()

            start_preprocess = time.time()

            get_sample_time += (start_preprocess - start_time) 

            for idx in range(len(sample['rgb'])):
                # ProjectPointCloud in RT-pose

                real_shape = [sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2], sample['rgb'][idx].shape[0]]
                # 370 * 1226 * 3
                sample['point_cloud'][idx] = sample['point_cloud'][idx].cuda()
                pcl = sample['point_cloud'][idx].clone()

                reflectance = None
                if _config['use_reflectance']:
                    reflectance = sample['reflectance'][idx].cuda()

                rotate_start_time = time.time()                
                R = mathutils.Quaternion(sample['rot_error'][idx]).to_matrix()
                R.resize_4x4()
                T = mathutils.Matrix.Translation(sample['tr_error'][idx])
                RT = T * R
                pc_rotated = rotate_back(pcl, RT)

                if _config['max_depth'] < 100.:
                    pc_rotated = pc_rotated[:, pc_rotated[0, :] < _config['max_depth']].clone()
                # project_start_time = time.time()
                # rotate_time += (project_start_time - rotate_start_time)
                cam_params = sample['calib'][idx].cuda()
                cam_model = CameraModel()
                cam_model.focal_length = cam_params[:2]
                cam_model.principal_point = cam_params[2:]
                uv, uvt, depth, dt, py, px, _ = cam_model.project_pytorch(pc_rotated, pcl, real_shape, reflectance)
                # occlusion_start_time = time.time()
                # project_time += (occlusion_start_time - project_start_time)

                uv = uv.t().int()
                uvt = uvt.t().int()

                depth_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                depth_img += 1000.
                depth_imgt = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                depth_imgt += 1000.
                depth_imgt = visibility.depth_image(uvt.contiguous(), dt, depth_imgt, uvt.shape[0], real_shape[1], real_shape[0])
                depth_img = visibility.depth_image(uv.contiguous(), depth, depth_img, uv.shape[0], real_shape[1], real_shape[0])
                depth_imgt[depth_imgt == 1000.] = 0.                              # num           , width        , height
                depth_img[depth_img == 1000.] = 0.

                # depth_img_no_occlusion = torch.zeros_like(depth_img, device='cuda')
                # depth_img_no_occlusion = visibility.visibility2(depth_img, cam_params, depth_img_no_occlusion,
                #                                                 depth_img.shape[1], depth_img.shape[0],
                #                                                 occlusion_threshold, _config['occlusion_kernel'])

                uv = uv.long()
                uvt = uvt.long()
                indexes = depth_img[uv[:,1], uv[:,0]] == depth
                flag = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                flagt = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                flag[uv[indexes,1], uv[indexes,0]] = 1
                flagt[uvt[indexes,1], uvt[indexes,0]] = 1
                depth_imgt = flagt * depth_imgt

                    # savepcl
                lidar_x = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                lidar_y = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                lidar_z = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                lidar_x[uvt[indexes,1], uvt[indexes,0]] = dt[indexes]
                lidar_y[uvt[indexes,1], uvt[indexes,0]] = py[indexes]
                lidar_z[uvt[indexes,1], uvt[indexes,0]] = px[indexes]
                
                # du = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                # du[uv[indexes,1], uv[indexes,0]] = (uvt[indexes,1] - uv[indexes,1]).float()
                # dv = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                # dv[uv[indexes,1], uv[indexes,0]] = (uvt[indexes,0] - uv[indexes,0]).float()
                # duv = torch.stack((du, dv))


                lidarToLidaru = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                lidarToLidaru[uvt[indexes,1], uvt[indexes,0]] = uv[indexes,1].float()
                lidarToLidarv = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                lidarToLidarv[uvt[indexes,1], uvt[indexes,0]] = uv[indexes,0].float()


                if _config['use_reflectance']:
                    # This need to be checked
                    uv = uv.long()
                    indexes = depth_img[uv[:,1], uv[:,0]] == depth
                    refl_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                    refl_img[uv[indexes,1], uv[indexes,0]] = refl[0, indexes]

                depth_img /= _config['max_depth']
                depth_imgt/= _config['max_depth']
                if not _config['use_reflectance']:
                    depth_img = depth_img.unsqueeze(0) # 于 0th 增加一个维度
                    depth_imgt = depth_imgt.unsqueeze(0)
                    flagt = flagt.unsqueeze(0)
                    flag = torch.stack((flag, flag))
                    lidarToLidaru = lidarToLidaru.unsqueeze(0)
                    lidarToLidarv = lidarToLidarv.unsqueeze(0)
                    pcl = torch.stack((lidar_x, lidar_y, lidar_z))
                else:
                    depth_img = torch.stack((depth_img, refl_img))
                # print(depth_img_no_occlusion.shape)
                # PAD ONLY ON RIGHT AND BOTTOM SIDE
                rgb = sample['rgb'][idx].cuda()



                shape_pad = [0, 0, 0, 0]

                # print('before pad rgb:{}'.format(rgb.shape)) # 3 * 370 * 1226
                # print('before pad lidar:{}'.format(depth_img_no_occlusion.shape)) # 1 * 370 * 1226

                shape_pad[3] = (img_shape[0] - rgb.shape[1])  # // 2
                shape_pad[1] = (img_shape[1] - rgb.shape[2])  # // 2 + 1

                # print('image_shape:{}'.format(img_shape)) # 384 * 1280
                # print('shape_pad:{}'.format(shape_pad)) # 0 54 0 14

                rgb = F.pad(rgb, shape_pad)
                depth_img = F.pad(depth_img, shape_pad)
                flagt = F.pad(flagt, shape_pad)
                flag = F.pad(flag, shape_pad)
                depth_imgt = F.pad(depth_imgt, shape_pad)
                pcl = F.pad(pcl, shape_pad)
                # duv = F.pad(duv, shape_pad)
                lidarToLidaru = F.pad(lidarToLidaru, shape_pad)
                lidarToLidarv = F.pad(lidarToLidarv, shape_pad)
                # p_uv = torch.zeros((2,rgb.shape[1],rgb.shape[2]), device='cuda', dtype=torch.float)



                if datasetType == 0:
                    info = torch.tensor([int(sample['idx'][idx]), int(sample['rgb_name'][idx])])
                elif datasetType == 1:
                    info = [sample['idx'][idx],sample['sub_dir'][idx], sample['rgb_name'][idx]]

                rgb_input.append(rgb)
                lidar_input.append(depth_img)
                flag_input.append(flag)
                flagt_input.append(flagt)
                depth_input.append(depth_imgt)
                # duv_input.append(duv)
                ltolu_input.append(lidarToLidaru)
                ltolv_input.append(lidarToLidarv)
                pcl_input.append(pcl)
                info_input.append(info)

            lidar_input = torch.stack(lidar_input)
            rgb_input = torch.stack(rgb_input)
            flag_input = torch.stack(flag_input)
            flagt_input = torch.stack(flagt_input)
            depth_input = torch.stack(depth_input)
            # duv_input = torch.stack(duv_input)
            ltolu_input = torch.stack(ltolu_input)
            ltolv_input = torch.stack(ltolv_input)
            pcl_input = torch.stack(pcl_input)
            # info_input = torch.stack(info_input)
            # pos_uv_input = torch.stack(pos_uv_input)
            # pos_xy_input = torch.stack(pos_xy_input)


            end_preprocess = time.time()
            # preprocess_time = (end_preprocess - start_preprocess)
            # pos_uv_input=0
            # pos_xy_input=0
            # print(end_preprocess - start_time)
            loss , trasl_e, rot_e= test(model, pcl_input, info_input, ltolu_input, ltolv_input, rgb_input, lidar_input, flagt_input, depth_input, sample['tr_error'],
                                        sample['rot_error'], loss_fn, dataset_val.model, sample['point_cloud'])

            # loss = train(model, optimizer, pcl_input, info_input, ltolu_input, ltolv_input, rgb_input, lidar_input, flagt_input, depth_input, sample['tr_error'],
            #                             sample['rot_error'], loss_fn, dataset_val.model, sample['point_cloud'])
            # # #
            # break
            if loss != loss:
                raise ValueError("Loss is NaN")
            train_time += (time.time() - end_preprocess)
            # print(f'{time.time() - start_time:.4f}')
            # train_writer.add_scalar("Loss", loss, total_iter)
            local_loss += loss
            if batch_idx % 50 == 0 and batch_idx != 0:

                print(f'Iter {batch_idx}/{len(TrainImgLoader)} training loss = {local_loss/50:.3f}, '
                      f'time = {(time.time() - start_time)/lidar_input.shape[0]:.4f}, '
                      #f'time_preprocess = {(end_preprocess-start_preprocess)/lidar_input.shape[0]:.4f}, '
                      f'time for 50 iter: {time.time()-time_for_50ep:.4f}')
                time_for_50ep = time.time()
                _run.log_scalar("Loss", local_loss/50, total_iter)
                local_loss = 0.
            total_train_loss += loss * len(sample['rgb'])
            total_iter += len(sample['rgb'])

            start_time = time.time()


        # total_time = time.time() - epoch_start_time
        # print("------------------------------------")
        # print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(dataset)))
        # print('Total epoch time = %.2f' % total_time)
        # print("------------------------------------")
        # _run.log_scalar("Total training loss", total_train_loss / len(dataset), epoch)

        if (epoch % 3 == 0) or (epoch > 115):
            ## Test ##
            total_test_loss = 0.
            total_test_t = 0.
            total_test_r = 0.

            local_loss = 0.0
            for batch_idx, sample in enumerate(TestImgLoader):
                # print(f'batch {batch_idx + 1}/{len(TestImgLoader)}', end='\r')
                start_time = time.time()
                lidar_input = []
                rgb_input = []
                duv_input = []
                depth_input = []
                flag_input = []
                flagt_input = []
                ltolu_input = []
                ltolv_input = []
                pcl_input = []
                info_input = []
                # pos_uv_input = []
                # pos_xy_input = []
                # print(time.time() - start_time)

                sample['tr_error'] = sample['tr_error'].cuda()
                sample['rot_error'] = sample['rot_error'].cuda()

                start_preprocess = time.time()

                get_sample_time += (start_preprocess - start_time)

                for idx in range(len(sample['rgb'])):
                    # ProjectPointCloud in RT-pose

                    real_shape = [sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2], sample['rgb'][idx].shape[0]]
                    # 370 * 1226 * 3
                    sample['point_cloud'][idx] = sample['point_cloud'][idx].cuda()
                    pcl = sample['point_cloud'][idx].clone()

                    reflectance = None
                    if _config['use_reflectance']:
                        reflectance = sample['reflectance'][idx].cuda()

                    rotate_start_time = time.time()
                    R = mathutils.Quaternion(sample['rot_error'][idx]).to_matrix()
                    R.resize_4x4()
                    T = mathutils.Matrix.Translation(sample['tr_error'][idx])
                    RT = T * R
                    pc_rotated = rotate_back(pcl, RT)

                    if _config['max_depth'] < 100.:
                        pc_rotated = pc_rotated[:, pc_rotated[0, :] < _config['max_depth']].clone()
                    # project_start_time = time.time()
                    # rotate_time += (project_start_time - rotate_start_time)
                    cam_params = sample['calib'][idx].cuda()
                    cam_model = CameraModel()
                    cam_model.focal_length = cam_params[:2]
                    cam_model.principal_point = cam_params[2:]
                    uv, uvt, depth, dt, py, px, _ = cam_model.project_pytorch(pc_rotated, pcl, real_shape, reflectance)
                    # occlusion_start_time = time.time()
                    # project_time += (occlusion_start_time - project_start_time)

                    uv = uv.t().int()
                    uvt = uvt.t().int()

                    depth_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                    depth_img += 1000.
                    depth_imgt = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                    depth_imgt += 1000.
                    depth_imgt = visibility.depth_image(uvt.contiguous(), dt, depth_imgt, uvt.shape[0], real_shape[1], real_shape[0])
                    depth_img = visibility.depth_image(uv.contiguous(), depth, depth_img, uv.shape[0], real_shape[1], real_shape[0])
                    depth_imgt[depth_imgt == 1000.] = 0.                              # num           , width        , height
                    depth_img[depth_img == 1000.] = 0.

                    # depth_img_no_occlusion = torch.zeros_like(depth_img, device='cuda')
                    # depth_img_no_occlusion = visibility.visibility2(depth_img, cam_params, depth_img_no_occlusion,
                    #                                                 depth_img.shape[1], depth_img.shape[0],
                    #                                                 occlusion_threshold, _config['occlusion_kernel'])

                    uv = uv.long()
                    uvt = uvt.long()
                    indexes = depth_img[uv[:,1], uv[:,0]] == depth
                    flag = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                    flagt = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                    flag[uv[indexes,1], uv[indexes,0]] = 1
                    flagt[uvt[indexes,1], uvt[indexes,0]] = 1
                    depth_imgt = flagt * depth_imgt

                    # savepcl
                    lidar_x = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                    lidar_y = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                    lidar_z = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                    lidar_x[uvt[indexes,1], uvt[indexes,0]] = dt[indexes]
                    lidar_y[uvt[indexes,1], uvt[indexes,0]] = py[indexes]
                    lidar_z[uvt[indexes,1], uvt[indexes,0]] = px[indexes]
                
                    # du = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                    # du[uv[indexes,1], uv[indexes,0]] = (uvt[indexes,1] - uv[indexes,1]).float()
                    # dv = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                    # dv[uv[indexes,1], uv[indexes,0]] = (uvt[indexes,0] - uv[indexes,0]).float()
                    # duv = torch.stack((du, dv))
                    lidarToLidaru = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                    lidarToLidaru[uvt[indexes,1], uvt[indexes,0]] = uv[indexes,1].float()
                    lidarToLidarv = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                    lidarToLidarv[uvt[indexes,1], uvt[indexes,0]] = uv[indexes,0].float()

                    if _config['use_reflectance']:
                        # This need to be checked
                        uv = uv.long()
                        indexes = depth_img[uv[:,1], uv[:,0]] == depth
                        refl_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                        refl_img[uv[indexes,1], uv[indexes,0]] = refl[0, indexes]

                    depth_img /= _config['max_depth']
                    depth_imgt/= _config['max_depth']
                    if not _config['use_reflectance']:
                        depth_img = depth_img.unsqueeze(0) # 于 0th 增加一个维度
                        depth_imgt = depth_imgt.unsqueeze(0)
                        flagt = flagt.unsqueeze(0)
                        lidarToLidaru = lidarToLidaru.unsqueeze(0)
                        lidarToLidarv = lidarToLidarv.unsqueeze(0)
                        flag = torch.stack((flag, flag))
                        pcl = torch.stack((lidar_x, lidar_y, lidar_z))
                    else:
                        depth_img = torch.stack((depth_img, refl_img))
                        # print(depth_img_no_occlusion.shape)
                    # PAD ONLY ON RIGHT AND BOTTOM SIDE
                    rgb = sample['rgb'][idx].cuda()



                    shape_pad = [0, 0, 0, 0]

                    # print('before pad rgb:{}'.format(rgb.shape)) # 3 * 370 * 1226
                    # print('before pad lidar:{}'.format(depth_img_no_occlusion.shape)) # 1 * 370 * 1226

                    shape_pad[3] = (img_shape[0] - rgb.shape[1])  # // 2
                    shape_pad[1] = (img_shape[1] - rgb.shape[2])  # // 2 + 1

                    # print('image_shape:{}'.format(img_shape)) # 384 * 1280
                    # print('shape_pad:{}'.format(shape_pad)) # 0 54 0 14

                    rgb = F.pad(rgb, shape_pad)
                    depth_img = F.pad(depth_img, shape_pad)
                    flagt = F.pad(flagt, shape_pad)
                    flag = F.pad(flag, shape_pad)
                    depth_imgt = F.pad(depth_imgt, shape_pad)
                    pcl = F.pad(pcl, shape_pad)
                    # duv = F.pad(duv, shape_pad)
                    lidarToLidaru = F.pad(lidarToLidaru, shape_pad)
                    lidarToLidarv = F.pad(lidarToLidarv, shape_pad)

                    # dep = depth_imgt.cpu()
                    # dep = np.array(dep)
                    # np.savetxt(r'course.txt', dep[0])
                    if datasetType == 0:
                        info = torch.tensor([int(sample['idx'][idx]), int(sample['rgb_name'][idx])])
                    elif datasetType == 1:
                        info = [sample['idx'][idx],sample['sub_dir'][idx], sample['rgb_name'][idx]]
                        
                    info_input.append(info)
                    rgb_input.append(rgb)
                    lidar_input.append(depth_img)
                    flag_input.append(flag)
                    flagt_input.append(flagt)
                    depth_input.append(depth_imgt)
                    ltolu_input.append(lidarToLidaru)
                    ltolv_input.append(lidarToLidarv)
                    pcl_input.append(pcl)
                    # break
                    # duv_input.append(duv)

                lidar_input = torch.stack(lidar_input)
                rgb_input = torch.stack(rgb_input)
                flag_input = torch.stack(flag_input)
                flagt_input = torch.stack(flagt_input)
                depth_input = torch.stack(depth_input)
                ltolu_input = torch.stack(ltolu_input)
                ltolv_input = torch.stack(ltolv_input)
                pcl_input = torch.stack(pcl_input)
                # info_input = torch.stack(info_input)
                # duv_input = torch.stack(duv_input)
                # pos_uv_input = torch.stack(pos_uv_input)
                # pos_xy_input = torch.stack(pos_xy_input)


                end_preprocess = time.time()

                loss , trasl_e, rot_e= test(model, pcl_input, info_input, ltolu_input, ltolv_input, rgb_input, lidar_input, flagt_input, depth_input, sample['tr_error'],
                                            sample['rot_error'], loss_fn, dataset_val.model, sample['point_cloud'])
                #
                if loss != loss:
                    raise ValueError("Loss is NaN")

                total_test_t += trasl_e
                total_test_r += rot_e
                local_loss += loss

                if batch_idx % 50 == 0 and batch_idx != 0:
                    print('Iter %d test loss = %.3f , time = %.2f' % (batch_idx, local_loss/50.,
                                                                      (time.time() - start_time)/lidar_input.shape[0]))
                    local_loss = 0.0
                total_test_loss += loss * len(sample['rgb'])
                # break

            print("------------------------------------")
            print('total test loss = %.3f' % (total_test_loss / len(dataset_val)))
            print(f'total traslation error: {total_test_t / len(dataset_val)} cm')
            print(f'total rotation error: {total_test_r / len(dataset_val)} °')
            # print(f'mean flow error: {flow_e}')
            print("------------------------------------")

            #train_writer.add_scalar("Val_Loss", total_test_loss / len(dataset_val), epoch)
            #train_writer.add_scalar("Val_t_error", total_test_t / len(dataset_val), epoch)
            #train_writer.add_scalar("Val_r_error", total_test_r / len(dataset_val), epoch)
            _run.log_scalar("Val_Loss", total_test_loss / len(dataset_val), epoch)
            _run.log_scalar("Val_t_error", total_test_t / len(dataset_val), epoch)
            _run.log_scalar("Val_r_error", total_test_r / len(dataset_val), epoch)

            # ## SAVE
            # val_loss = total_test_loss / len(dataset_val)
            # if val_loss < BEST_VAL_LOSS:
            #     BEST_VAL_LOSS = val_loss
            #     #_run.result = BEST_VAL_LOSS
            #     if _config['rescale_transl'] > 0:
            #         _run.result = total_test_t / len(dataset_val)
            #     else:
            #         _run.result = total_test_r / len(dataset_val)
            #     savefilename = f'{_config["savemodel"]}/checkpoint_r{_config["max_r"]:.2f}_t{_config["max_t"]:.2f}_e{epoch}_{val_loss:.3f}_H8k10-1.tar'
            #     torch.save({
            #         'config': _config,
            #         'epoch': epoch,
            #         'state_dict': model.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #         'train_loss': total_train_loss / len(dataset),
            #         'test_loss': total_test_loss / len(dataset_val),
            #     }, savefilename)
            #     print(f'Model saved as {savefilename}')
            #     if old_save_filename is not None:
            #         if os.path.exists(old_save_filename):
            #             os.remove(old_save_filename)
            #     old_save_filename = savefilename
        break
    print('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))
    return _run.result
