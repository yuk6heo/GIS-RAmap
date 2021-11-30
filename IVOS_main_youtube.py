from davisinteractive.dataset import Davis

from libs import custom_transforms as tr
import os

import numpy as np
import json
from PIL import Image
import csv
from datetime import datetime
import cv2

import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from libs import utils_custom, utils_visualize
from config import Config
from networks.network import NET_GAmap
from libs.youtubesession import YoutubeSession
from datasets_torch.torchdatasets_youtube_test import YoutubeVOS



class Main_tester(object):
    def __init__(self, config,  n_total_rounds=4):
        self.config = config
        self.Davisclass = Davis(self.config.davis_dataset_dir)
        self.current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        self._palette = Image.open(self.config.palette_dir).getpalette()
        self.save_res_dir = str()
        self.save_log_dir = str()
        self.save_logger = None
        self.save_csvsummary_dir = str()

        self.net = NET_GAmap()
        self.net.cuda()
        self.net.eval()

        self.net.load_state_dict(torch.load('checkpoints/ckpt_wo_YTIVOStrainset.pth'))

        self.max_nb_interactions = 4

        self.img_size, self.num_frames, self.n_objects, self.final_masks, self.tmpdict_siact = None, None, None, None, None
        self.pad_info, self.hpad1, self.wpad1, self.hpad2, self.wpad2 = None, None, None, None, None
        self.n_total_rounds = n_total_rounds

    def run_youtube(self, n_ipoint, save_res_dir):

        self.n_init_points = n_ipoint
        test_load_points_dir = 'etc/0youtube_iPoints/point_info_{:02d}.json'.format(n_ipoint)
        self.session = YoutubeSession(self.config, n_total_rounds=self.n_total_rounds, test_load_points_dir= test_load_points_dir)

        self.save_res_dir = save_res_dir + '/iPoint{:02d}'.format(n_ipoint)
        utils_custom.mkdir(self.save_res_dir)
        self.save_csvsummary_dir = os.path.join(self.save_res_dir, 'summary_in_csv.csv')
        self.save_log_dir = os.path.join(self.save_res_dir, 'test_logs.txt')
        self.save_logger = utils_custom.logger(self.save_log_dir)
        self.save_logger.printNlog(dir_name)

        with torch.no_grad():
            pf = self.run_IVOS()
        return pf

    def run_IVOS(self):
        seen_seq = {}
        output_dict = dict()
        output_dict['average_objs_iou'] = dict()
        output_dict['average_iact_iou'] = np.zeros(self.max_nb_interactions)
        output_dict['annotated_frames'] = dict()

        with open(self.save_csvsummary_dir, mode='a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['sequence', 'obj_idx'] + ['round-' + str(i + 1) for i in range(self.max_nb_interactions)])


        for ii,video in enumerate(self.session.youtube_val_videos):
            print('{:04d}th-video'.format(ii+1))
            self.img_size, self.num_frames, self.n_objects = \
                self.session.initialize_session_for_video(video)

            self.sequence = video
            anno_dict = {'frames': [], 'annotated_masks': [], 'masks_tobe_modified': []}
            seen_seq[self.sequence] = 1 if self.sequence not in seen_seq.keys() else seen_seq[self.sequence] + 1
            scr_id = seen_seq[self.sequence]
            self.final_masks = np.zeros([self.num_frames, self.img_size[0], self.img_size[1]])
            pd_r_img, self.pad_info = utils_custom.apply_pad(self.resize_shorter480_seg(self.final_masks[0]))
            self.hpad1, self.wpad1 = self.pad_info[0][0], self.pad_info[1][0]
            self.hpad2, self.wpad2 = self.pad_info[0][1], self.pad_info[1][1]
            self.prob_map_of_frames = torch.zeros((self.num_frames, self.n_objects + 1, pd_r_img.shape[0], pd_r_img.shape[1])).cuda()
            self.anno_6chEnc_r4_list = []
            self.anno_3chEnc_r4_list = []
            self.scores_ni_nf = np.zeros([8, self.num_frames])
            IoU_over_eobj = []


            for n_interaction in range(1, self.n_total_rounds+1):

                annotated_now = self.session.prev_frames_used_in_index[-1]
                anno_dict['frames'].append(annotated_now)  # Where we save annotated frames
                anno_dict['masks_tobe_modified'].append(self.final_masks[annotated_now])  # mask before modefied at the annotated frame

                if n_interaction == 1:
                    data_points_or_scr = self.session.get_points_firstround()
                else:
                    data_points_or_scr = self.session.get_scrdata_currentround()

                self.save_logger.printNlog('\nRunning sequence {} in round: {}'.format(self.sequence, n_interaction))

                # Get Predicted mask & Mask decision from pred_mask
                self.final_masks = self.run_VOS_singleiact(n_interaction, data_points_or_scr, anno_dict['frames'])  # self.final_mask changes

                # Limit candidate frames
                if n_interaction != self.max_nb_interactions:
                    self.scores_ni_nf[n_interaction] = self.scores_ni_nf[n_interaction-1]
                current_score_np = self.scores_ni_nf[n_interaction-1]

                if self.config.test_guide_method=='RS1':
                    next_scribble_frame_candidates = list(np.argsort(current_score_np)[:1])
                elif self.config.test_guide_method=='RS4':
                    sorted_score_idx = np.argsort(current_score_np)
                    exclude_range = self.num_frames/10
                    excluded_next_candidates = []
                    next_scribble_frame_candidates = []
                    for i in range(self.num_frames):
                        if not sorted_score_idx[i] in excluded_next_candidates:
                            next_scribble_frame_candidates.append(sorted_score_idx[i])
                            excluded_next_candidates += list(range(
                                int(sorted_score_idx[i]-(exclude_range/2)+0.5), int(sorted_score_idx[i]+(exclude_range/2)+0.5)))
                        if len(next_scribble_frame_candidates)==4:
                            break
                elif self.config.test_guide_method=='wo_RS':
                    next_scribble_frame_candidates=None
                else:
                    raise NotImplementedError

                # Submit your prediction
                self.session.submit_masks(self.final_masks, next_scribble_frame_candidates)  # F, H, W

                if self.config.test_save_pngs_option:
                    utils_custom.mkdir(os.path.join(self.save_res_dir, 'result_video', '{}/round{:02d}'.format(self.sequence, n_interaction)))
                    for fr_idx in range(self.num_frames):
                        savefname = os.path.join(self.save_res_dir, 'result_video','{}/round{:02d}'.format(self.sequence, n_interaction),self.session.frame_list[fr_idx])
                        tmpPIL = Image.fromarray(self.final_masks[fr_idx].astype(np.uint8), 'P')
                        tmpPIL.putpalette(self._palette)
                        tmpPIL.save(savefname)

                ## Visualizers and Saver
                # IoU estimation
                IoU_over_eobj.append(self.session.metric)

                # save final mask in anno_dict
                anno_dict['annotated_masks'].append(self.final_masks[annotated_now])  # mask after modefied at the annotated frame

                if n_interaction == 4:  # After Lastround -> total 90 iter
                    # IoU manager
                    IoU_over_eobj = np.stack(IoU_over_eobj, axis=0)  # niact,frames,n_obj
                    IoUeveryround_perobj = np.mean(IoU_over_eobj, axis=1)  # niact,n_obj
                    output_dict['average_iact_iou'] += np.sum(IoU_over_eobj[list(range(n_interaction)), anno_dict['frames']], axis=-1)
                    output_dict['annotated_frames'][self.sequence] = anno_dict['frames']
                    if self.config.test_save_pngs_option:
                        savefiledir = os.path.join(self.save_res_dir, 'plot_IoU_perObj')
                        utils_custom.mkdir(savefiledir)
                        for obj_idx in range(self.n_objects):
                            savefilename = os.path.join(savefiledir, self.sequence + '-obj' + str(obj_idx + 1) + '_first{:03d}final{:03d}.png'
                                                        .format(int(1000 * IoUeveryround_perobj[0, obj_idx]),
                                                                int(1000 * IoUeveryround_perobj[-1, obj_idx])))
                            utils_visualize.visualize_interactionIoU(IoU_over_eobj[:, :, obj_idx], self.sequence + '-obj' + str(obj_idx + 1),
                                                                     anno_dict['frames'], save_dir=savefilename, show_propagated_region=True)

                    # write csv
                    for obj_idx in range(self.n_objects):
                        with open(self.save_csvsummary_dir, mode='a') as csv_file:
                            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            writer.writerow([self.sequence, str(obj_idx + 1)] + list(IoUeveryround_perobj[:, obj_idx]))

        savefname = self.save_res_dir + '/summary.json'
        performanceJF = list(self.session.performance_summary())
        with open(savefname, "w") as json_file:
            json.dump(performanceJF, json_file)

        torch.cuda.empty_cache()
        model = None
        self.save_logger.printNlog(str(performanceJF))
        return performanceJF

    def run_VOS_singleiact(self, n_interaction, data_points_or_scr, annotated_frames):
        '''
        pm_ps_ns => # n_obj,3,h,w
        '''

        annotated_frames_np = np.array(annotated_frames)
        annotated_now = annotated_frames[-1]

        prop_list = utils_custom.get_prop_list(annotated_frames, annotated_now, self.num_frames, proportion=self.config.test_propagation_proportion)

        if n_interaction == 1:
            pm_ps_ns_3ch_np = self.point_data_to_img(data_points_or_scr)
        else:
            pm_ps_ns_3ch_np = self.scr_data_to_img(data_points_or_scr, n_interaction, annotated_now)
        pm_ps_ns_3ch_t = torch.from_numpy(pm_ps_ns_3ch_np).cuda()

        if (prop_list[0] != annotated_now) and (prop_list.count(annotated_now) != 2):
            print(str(prop_list))
            raise NotImplementedError
        print(str(prop_list))  # we made our proplist first backward, and then forward

        composed_transforms = transforms.Compose([tr.Normalize_ApplymeanvarImage(self.config.mean, self.config.var),
                                                  tr.ToTensor()])
        db_test = YoutubeVOS(self.config.youtube_dataset_dir, self.config, transform=composed_transforms,  custom_frames=prop_list,
                             seq_name=self.sequence, resize=True,)
        testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=self.config.test_n_workers, pin_memory=True)

        flag = 0  # 1: propagating backward, 2: propagating forward
        print('[{:01d} round] processing...'.format(n_interaction))

        for ii, batched in enumerate(testloader):
            # batched : image, scr_img, 0~fr, meta
            inpdict = dict()
            operating_frame = int(batched['meta']['frame_id'][0])

            for inp in batched:
                if inp == 'meta': continue
                inpdict[inp] = Variable(batched[inp]).cuda()

            inpdict['image'] = inpdict['image'].expand(self.n_objects, -1, -1, -1)

            #################### Iaction ########################
            if operating_frame == annotated_now:  # Check the round is on interaction
                if flag == 0:
                    flag += 1
                    adjacent_to_anno = True
                elif flag == 1:
                    flag += 1
                    adjacent_to_anno = True
                    continue
                else:
                    raise NotImplementedError

                pm_ps_ns_3ch_t = torch.nn.ReflectionPad2d(self.pad_info[1] + self.pad_info[0])(pm_ps_ns_3ch_t)
                inputs = torch.cat([inpdict['image'], pm_ps_ns_3ch_t], dim=1)
                anno_3chEnc_r4, _ = self.net.encoder_3ch.forward(inpdict['image'])
                neighbor_pred_onehot_sal, anno_6chEnc_r4 = self.net.forward_obj_feature_extractor(inputs)  # [nobj, 1, P_H, P_W], # [n_obj,2048,h/16,w/16]

                output_logit, r4_anno, score = self.net.forward_prop(
                    [anno_3chEnc_r4], inpdict['image'], [anno_6chEnc_r4],
                    anno_3chEnc_r4, torch.sigmoid(neighbor_pred_onehot_sal),
                    anno_fr_list= annotated_frames_np, que_fr= operating_frame)  # [nobj, 1, P_H, P_W]

                output_prob_tmp = F.softmax(output_logit, dim=1) # [nobj, 2, P_H, P_W]
                output_prob_tmp = output_prob_tmp[:, 1] # [nobj, P_H, P_W]
                one_hot_outputs_t = F.softmax(self.soft_aggregation(output_prob_tmp), dim=0) # [nobj+1, P_H, P_W]
                anno_onehot_prob = one_hot_outputs_t.clone()[1:].unsqueeze(1) # [nobj, 1, P_H, P_W]


                anno_3chEnc_r4, r2_prev_fromanno = self.net.encoder_3ch.forward(inpdict['image'])
                self.anno_6chEnc_r4_list.append(anno_6chEnc_r4)
                self.anno_3chEnc_r4_list.append(anno_3chEnc_r4)

                if len(self.anno_6chEnc_r4_list) != len(annotated_frames):
                    raise NotImplementedError

            #################### Propagation ########################
            else:
                # Flag [1: propagating backward, 2: propagating forward]
                if adjacent_to_anno:
                    r4_neighbor = r4_anno
                    neighbor_pred_onehot = anno_onehot_prob
                else:
                    r4_neighbor = r4_que
                    neighbor_pred_onehot = targ_onehot_prob
                adjacent_to_anno = False

                output_logit, r4_que, score = self.net.forward_prop(
                    self.anno_3chEnc_r4_list, inpdict['image'], self.anno_6chEnc_r4_list,
                    r4_neighbor, neighbor_pred_onehot,
                    anno_fr_list= annotated_frames_np, que_fr= operating_frame)  # [nobj, 1, P_H, P_W]
                output_prob_tmp = F.softmax(output_logit, dim=1) # [nobj, 2, P_H, P_W]
                output_prob_tmp = output_prob_tmp[:, 1] # [nobj, P_H, P_W]
                one_hot_outputs_t = F.softmax(self.soft_aggregation(output_prob_tmp), dim=0) # [nobj+1, P_H, P_W]

                targ_onehot_prob = one_hot_outputs_t.clone()[1:].unsqueeze(1) # [nobj, 1, P_H, P_W]

            # Final mask indexing
            self.prob_map_of_frames[operating_frame] = one_hot_outputs_t
            onehot_out_tmp = F.interpolate(
                one_hot_outputs_t[:,self.hpad1:-self.hpad2, self.wpad1:-self.wpad2].unsqueeze(dim=0), size=self.final_masks[0].shape)
            self.final_masks[operating_frame] = torch.argmax(onehot_out_tmp[0],dim=0).cpu().numpy().astype(np.uint8)
            self.scores_ni_nf[n_interaction-1,operating_frame] = score

        torch.cuda.empty_cache()

        return self.final_masks


    def soft_aggregation(self, ps):
        num_objects, H, W = ps.shape
        em = torch.zeros(num_objects +1, H, W).cuda()
        em[0] =  torch.prod(1-ps, dim=0) # bg prob
        em[1:num_objects+1] = ps # obj prob
        em = torch.clamp(em, 1e-7, 1-1e-7)
        logit = torch.log((em /(1-em)))
        return logit


    def resize_shorter480(self, img,seg):
        ori_h, ori_w = img.shape[0], img.shape[1]
        if ori_w >= ori_h:
            if ori_h ==480:
                return img,seg
            new_h = 480
            new_w = int((ori_w / ori_h) * 480)
        else:
            if ori_w ==480:
                return img,seg
            new_w = 480
            new_h = int((ori_h / ori_w) * 480)

        output_size = (new_w, new_h)

        new_img = cv2.resize(img, output_size, interpolation=cv2.INTER_CUBIC)
        new_seg = cv2.resize(seg, output_size, interpolation=cv2.INTER_NEAREST)
        return new_img, new_seg

    def resize_shorter480_seg(self, seg):
        ori_h, ori_w = seg.shape[0], seg.shape[1]
        if ori_w >= ori_h:
            new_h = 480
            new_w = int((ori_w / ori_h) * 480)
        else:
            new_w = 480
            new_h = int((ori_h / ori_w) * 480)

        output_size = (new_w, new_h)
        new_seg = cv2.resize(seg, output_size, interpolation=cv2.INTER_NEAREST)
        return new_seg


    def point_data_to_img(self, points_data):
        zeros_map = np.zeros_like(self.resize_shorter480_seg(self.final_masks[0]))
        ori_h, ori_w = self.final_masks[0].shape
        if ori_w >= ori_h: sizerate = 480 / ori_h
        else:              sizerate = 480 / ori_w
        try:points_data = (np.asarray(points_data)*sizerate).astype(np.int64)
        except:
            a=1

        pm_ps_ns_3ch_t = []  # n_obj,3,h,w
        for obj_id in range(1, self.n_objects + 1):
            ptmap_tmp = np.zeros_like(zeros_map, dtype=np.float32)
            ptmap_tmp[points_data[obj_id-1][0, :], points_data[obj_id-1][1, :]]=1
            pos_ptimg = utils_custom.scrimg_postprocess(ptmap_tmp, dilation=7, blur=True, blursize=(5, 5), var=6.0)
            pm_ps_ns_3ch_t.append(np.stack([np.ones_like(pos_ptimg) / 2, pos_ptimg, np.zeros_like(pos_ptimg)], axis=0))
        pm_ps_ns_3ch_t = np.stack(pm_ps_ns_3ch_t, axis=0)  # n_obj,3,h,w

        return pm_ps_ns_3ch_t

    def scr_data_to_img(self, scribbles_data, n_interaction, annotated_now):

        prev_mask = self.resize_shorter480_seg(self.final_masks[annotated_now])
        scr_data = scribbles_data['scribbles']

        # Interaction settings
        pm_ps_ns_3ch_t = []  # n_obj,3,h,w
        for obj_id in range(1, self.n_objects + 1):
            prev_round_input = (prev_mask == obj_id).astype(np.float32)  # H,W
            pos_scrimg, neg_scrimg = utils_custom.scribble_to_image(scr_data, annotated_now, obj_id,
                                                                    dilation=self.config.scribble_dilation_param,
                                                                    prev_mask=prev_mask, blur=True,
                                                                    singleimg=False, seperate_pos_neg=True)
            pm_ps_ns_3ch_t.append(np.stack([prev_round_input, pos_scrimg, neg_scrimg], axis=0))
        pm_ps_ns_3ch_t = np.stack(pm_ps_ns_3ch_t, axis=0)  # n_obj,3,h,w

        return pm_ps_ns_3ch_t


if __name__ == '__main__':
    config = Config()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.test_gpu_id)

    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    dir_name = os.path.split(os.path.split(__file__)[0])[1] + '[JF]_[' + config.test_guide_method + ']_' + current_time
    save_res_dir = os.path.join(config.test_result_yt_dir, dir_name)
    utils_custom.mkdir(save_res_dir)

    tester = Main_tester(config)
    performanceJF_05 = tester.run_youtube(5, save_res_dir)
    performanceJF_10 = tester.run_youtube(10, save_res_dir)
    performanceJF_20 = tester.run_youtube(20, save_res_dir)
    performanceJF_50 = tester.run_youtube(50, save_res_dir)

    savefname = save_res_dir + '/summary_total.json'
    performanceJF = np.array(performanceJF_05)+np.array(performanceJF_10)+np.array(performanceJF_20)+np.array(performanceJF_50)
    performanceJF = performanceJF.tolist()
    with open(savefname, "w") as json_file:
        json.dump(performanceJF, json_file)
