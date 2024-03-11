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

from camera_model_localization import CameraModel
from Dataset_kitti_localization import DatasetVisibilityKittiSingle

from losses import DistancePoints3D, GeometricLoss, L1Loss, ProposedLoss
# from models.CMRNet.CMRNet_norm import CMRNet
from models.CMRNet.CMRNet_single_loc import CMRNet
from quaternion_distances import quaternion_distance
from utils import merge_inputs, overlay_imgs, rotate_back


datasetType = 0 # 0 ---- kitti 1 ---- argo 9 -- mixed

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
    rescale_rot = 1
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


# CCN training
@ex.capture
def train(model, optimizer, rgb_img, refl_img, target_transl, target_rot, loss_fn, point_clouds, loss):

    optimizer.zero_grad()

    # Run model
    transl_err, rot_err = model(rgb_img, refl_img) # ,  w_x, w_q
    # total_loss = loss_fn(target_transl, target_rot, transl_err, rot_err)

    # loss_q1 = torch.mean(torch.sqrt(torch.sum((target_rot-rot_err)*(target_rot-rot_err), dim=-1, keepdim=True)+1e-10))
    # loss_t1 = torch.mean(torch.sqrt((target_transl-transl_err)*(target_transl-transl_err)+1e-10))
    # total_loss = loss_t1 * torch.exp(-w_x) + w_x + loss_q1 * torch.exp(-w_q) + w_q


    if loss != 'points_distance':
        total_loss = loss_fn(target_transl, target_rot, transl_err, rot_err)
        # total_loss2 = loss_fn(target_transl, target_rot, t0, r0)
    else:
        total_loss = loss_fn(point_clouds, target_transl, target_rot, transl_err, rot_err)
        # total_loss2 = loss_fn(target_transl, target_rot, t0, r0)
    # with apex.amp.scale_loss(total_loss, optimizer) as scaled_loss:
    #     scaled_loss.backward()
    # total_loss = 0.6*total_loss1 + 0.4*total_loss2
    total_loss.backward()
    optimizer.step()

    return total_loss.item()


# CNN test
@ex.capture
def test(model, rgb_img, refl_img,target_transl, target_rot, loss_fn, camera_model, point_clouds, loss):

    with torch.no_grad():
        transl_err, rot_err = model(rgb_img, refl_img) # ,  w_x, w_q



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
        elif datasetType == 9:
            _config['test_sequence'] = "mixed"
        print("Test Sequence: ", _config['test_sequence'])
        dataset_class = DatasetVisibilityKittiSingle
    occlusion_threshold = _config['occlusion_threshold']


    if datasetType == 0:
        img_shape = (384, 1280)
    elif datasetType == 1:
        img_shape = (640, 960) 
    elif datasetType == 9:
        img_shape = (640, 1280)



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
                                                 shuffle=True,
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
        
        cnt = 0
        for batch_idx, sample in enumerate(TrainImgLoader):
            # break

            # break
            # print(f'batch {batch_idx+1}/{len(TrainImgLoader)}', end='\r')

            lidar_input = []
            rgb_input = []


            sample['tr_error'] = sample['tr_error'].cuda()
            sample['rot_error'] = sample['rot_error'].cuda()

            start_preprocess = time.time()

            get_sample_time += (start_preprocess - start_time) 

            for idx in range(len(sample['rgb'])):

                # ProjectPointCloud in RT-pose
                cnt += 1
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

                uv, depth, px, py, refl = cam_model.project_pytorch(pc_rotated, real_shape, reflectance)
                # occlusion_start_time = time.time()
                # project_time += (occlusion_start_time - project_start_time)

                uv = uv.t().int()


                depth_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                depth_img += 1000.
                depth_img = visibility.depth_image(uv.contiguous(), depth, depth_img, uv.shape[0], real_shape[1], real_shape[0])                         # num           , width        , height
                depth_img[depth_img == 1000.] = 0.
                depth_img_no_occlusion = torch.zeros_like(depth_img, device='cuda')
                depth_img_no_occlusion = visibility.visibility2(depth_img, cam_params, depth_img_no_occlusion,
                                                                depth_img.shape[1], depth_img.shape[0],
                                                                occlusion_threshold, _config['occlusion_kernel'])




                if _config['use_reflectance']:
                    # This need to be checked
                    uv = uv.long()
                    indexes = depth_img[uv[:,1], uv[:,0]] == depth
                    refl_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                    refl_img[uv[indexes,1], uv[indexes,0]] = refl[0, indexes]

                depth_img /= _config['max_depth']

                if not _config['use_reflectance']:
                    depth_img = depth_img.unsqueeze(0) # 于 0th 增加一个维度
                else:
                    depth_img = torch.stack((depth_img, refl_img))
                

                rgb = sample['rgb'][idx].cuda()

                shape_pad = [0, 0, 0, 0]


                shape_pad[3] = (img_shape[0] - rgb.shape[1])  # // 2
                shape_pad[1] = (img_shape[1] - rgb.shape[2])  # // 2 + 1



                rgb = F.pad(rgb, shape_pad)
                
                depth_img = F.pad(depth_img, shape_pad)


                rgb_input.append(rgb)
                lidar_input.append(depth_img)




            lidar_input = torch.stack(lidar_input)
            rgb_input = torch.stack(rgb_input)



            end_preprocess = time.time()
            # preprocess_time = (end_preprocess - start_preprocess)


            loss = train(model, optimizer, rgb_input, lidar_input, sample['tr_error'],
                         sample['rot_error'], loss_fn, sample['point_cloud'])
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


        total_time = time.time() - epoch_start_time
        print("------------------------------------")
        print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(dataset)))
        print('Total epoch time = %.2f' % total_time)
        print("------------------------------------")
        _run.log_scalar("Total training loss", total_train_loss / len(dataset), epoch)
        cnt = 0
        if (epoch <= 120 and epoch % 20 == 0) or (epoch > 120 and epoch % 10 == 0) or (epoch > 195):
            ## Test ##
            total_test_loss = 0.
            total_test_t = 0.
            total_test_r = 0.
            
            local_loss = 0.0
            time_0 = time.time()
            
            
            
            for batch_idx, sample in enumerate(TestImgLoader):
                time_for_load = time.time()-time_0

                # print(f'batch {batch_idx + 1}/{len(TestImgLoader)}', end='\r')
                start_time = time.time()
                lidar_input = []
                rgb_input = []

                sample['tr_error'] = sample['tr_error'].cuda()
                sample['rot_error'] = sample['rot_error'].cuda()

                start_preprocess = time.time()
                time_for_rotation = 0
                time_for_project = 0
                time_for_depth = 0
                time_for_occlusion = 0
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
                    time_for_rotation += time.time() - rotate_start_time


                    project_start_time = time.time()
                    cam_params = sample['calib'][idx].cuda()
                    cam_model = CameraModel()
                    cam_model.focal_length = cam_params[:2]
                    cam_model.principal_point = cam_params[2:]

                    uv, depth, py, px, refl = cam_model.project_pytorch(pc_rotated , real_shape, reflectance)      
                    # occlusion_start_time = time.time()
                    # pryangyioject_time += (occlusion_start_time - project_start_time)
                    time_for_project += time.time() - project_start_time

                    depth_start_time = time.time()
                    uv = uv.t().int()
                    # print(uv.shape)

                    depth_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                    depth_img += 1000.
                    depth_img = visibility.depth_image(uv.contiguous(), depth, depth_img, uv.shape[0], real_shape[1], real_shape[0])                        # num           , width        , height
                    depth_img[depth_img == 1000.] = 0.
                    time_for_depth += time.time()-depth_start_time
                    # depth_img_no_occlusion = torch.zeros_like(depth_img, device='cuda')
                    # depth_img_no_occlusion = visibility.visibility2(depth_img, cam_params, depth_img_no_occlusion,
                    #                                                 depth_img.shape[1], depth_img.shape[0],
                    #                                                 occlusion_threshold, _config['occlusion_kernel'])


                    if _config['use_reflectance']:
                        # This need to be checked
                        uv = uv.long()
                        indexes = depth_img[uv[:,1], uv[:,0]] == depth
                        refl_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                        refl_img[uv[indexes,1], uv[indexes,0]] = refl[0, indexes]

                    depth_img /= _config['max_depth']

                    if not _config['use_reflectance']:
                        depth_img = depth_img.unsqueeze(0) # 
                    else:
                        depth_img = torch.stack((depth_img, refl_img))

                        # print(depth_img_no_occlusion.shape)
                    # PAD ONLY ON RIGHT AND BOTTOM SIDE
                    rgb = sample['rgb'][idx].cuda()



                    shape_pad = [0, 0, 0, 0]

                    shape_pad[3] = (img_shape[0] - rgb.shape[1])  # // 2
                    shape_pad[1] = (img_shape[1] - rgb.shape[2])  # // 2 + 1


                    rgb = F.pad(rgb, shape_pad)
                    depth_img = F.pad(depth_img, shape_pad)

                    rgb_input.append(rgb)
                    lidar_input.append(depth_img)

                    # break

                lidar_input = torch.stack(lidar_input)
                rgb_input = torch.stack(rgb_input)

 

                loss, trasl_e, rot_e = test(model,rgb_input, lidar_input, sample['tr_error'],
                                            sample['rot_error'], loss_fn, dataset_val.model, sample['point_cloud'])

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
                time_0 = time.time()
                # break

            print("------------------------------------")
            print('total test loss = %.3f' % (total_test_loss / len(dataset_val)))
            print(f'total traslation error: {total_test_t / len(dataset_val)} cm')
            print(f'total rotation error: {total_test_r / len(dataset_val)} °')

            print("------------------------------------")

            _run.log_scalar("Val_Loss", total_test_loss / len(dataset_val), epoch)
            _run.log_scalar("Val_t_error", total_test_t / len(dataset_val), epoch)
            _run.log_scalar("Val_r_error", total_test_r / len(dataset_val), epoch)

            # SAVE
            val_loss = total_test_loss / len(dataset_val)
            if val_loss < BEST_VAL_LOSS:
                BEST_VAL_LOSS = val_loss
                #_run.result = BEST_VAL_LOSS
                if _config['rescale_transl'] > 0:
                    _run.result = total_test_t / len(dataset_val)
                else:
                    _run.result = total_test_r / len(dataset_val)
                savefilename = f'{_config["savemodel"]}/checkpoint_r{_config["max_r"]:.2f}_t{_config["max_t"]:.2f}_e{epoch}_{val_loss:.3f}_SJTU_LOC.tar'
                torch.save({
                    'config': _config,
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'train_loss': total_train_loss / len(dataset),
                    'test_loss': total_test_loss / len(dataset_val),
                }, savefilename)
                print(f'Model saved as {savefilename}')
                if old_save_filename is not None:
                    if os.path.exists(old_save_filename):
                        os.remove(old_save_filename)
                old_save_filename = savefilename

    print('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))
    return _run.result

