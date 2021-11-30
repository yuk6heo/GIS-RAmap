from davisinteractive.session import DavisInteractiveSession
from davisinteractive import utils as interactive_utils
from davisinteractive.dataset import Davis
from davisinteractive.metrics import batched_jaccard, batched_f_measure

from libs import custom_transforms as tr
from datasets_torch import davis_2017
import os

import time
import numpy as np
import json
import pickle
from PIL import Image
import csv
from datetime import datetime

import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from libs import utils_custom, utils_visualize
from libs.analyze_report import analyze_summary
from config import Config
from networks.network import NET_GAmap
import warnings
warnings.filterwarnings("ignore")

class Main_tester(object):
    def __init__(self, config):
        self.config = config
        self.Davisclass = Davis(self.config.davis_dataset_dir)
        self.current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        self._palette = Image.open(self.config.palette_dir).getpalette()
        self.save_res_dir = str()
        self.save_log_dir = str()
        self.save_logger = None
        self.save_csvsummary_dir = str()
        self.n_operated_frames_accum = 0
        self.total_taken_times_accum = 0

        self.net = NET_GAmap()
        self.net.cuda()
        self.net.eval()

        self.net.load_state_dict(torch.load('checkpoints/ckpt_standard.pth'))
        self.scr_indices = [1, 2, 3]
        self.max_nb_interactions = 8
        self.max_time = self.max_nb_interactions * 30

        self.scr_samples = []
        for v in sorted(self.Davisclass.sets[self.config.test_subset]):
            for idx in self.scr_indices:
                self.scr_samples.append((v, idx))

        self.img_size, self.num_frames, self.n_objects, self.final_masks, self.tmpdict_siact = None, None, None, None, None
        self.pad_info, self.hpad1, self.wpad1, self.hpad2, self.wpad2 = None, None, None, None, None

    def run_for_diverse_metrics(self, ):
        with torch.no_grad():
            for metric in self.config.test_metric_list:
                if metric == 'J':
                    dir_name = os.path.split(os.path.split(__file__)[0])[1] + '[J]_['  + self.config.test_guide_method + ']_' + self.current_time
                elif metric == 'J_AND_F':
                    dir_name = os.path.split(os.path.split(__file__)[0])[1] + '[JF]_[' + self.config.test_guide_method + ']_' + self.current_time
                else:
                    dir_name = None
                    print("Impossible metric is contained in config.test_metric_list!")
                    raise NotImplementedError()
                self.save_res_dir = os.path.join(self.config.test_result_df_dir, dir_name)
                utils_custom.mkdir(self.save_res_dir)
                self.save_csvsummary_dir = os.path.join(self.save_res_dir, 'summary_in_csv.csv')
                self.save_log_dir = os.path.join(self.save_res_dir, 'test_logs.txt')
                self.save_logger = utils_custom.logger(self.save_log_dir)
                self.save_logger.printNlog(dir_name + self.current_time)

                self.run_IVOS(metric)

    def run_IVOS(self, metric):
        seen_seq = {}
        numseq, tmpseq = 0, ''

        with open(self.save_csvsummary_dir, mode='a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['sequence', 'obj_idx', 'scr_idx'] + ['round-' + str(i + 1) for i in range(self.max_nb_interactions)])

        with DavisInteractiveSession(host=self.config.test_host,
                                     user_key=self.config.test_userkey,
                                     davis_root=self.config.davis_dataset_dir,
                                     subset=self.config.test_subset,
                                     report_save_dir=self.save_res_dir,
                                     max_nb_interactions=self.max_nb_interactions,
                                     max_time=self.max_time,
                                     metric_to_optimize=metric) as sess:

            sess.connector.service.robot.min_nb_nodes = self.config.test_min_nb_nodes
            sess.samples = self.scr_samples

            while sess.next():
                # Get the current iteration scribbles
                self.sequence, scribbles, first_scribble = sess.get_scribbles(only_last=False)

                if first_scribble:
                    anno_dict = {'frames': [], 'annotated_masks': [], 'masks_tobe_modified': []}
                    n_interaction = 1
                    info = Davis.dataset[self.sequence]
                    self.img_size = info['image_size'][::-1]
                    self.num_frames = info['num_frames']
                    self.n_objects = info['num_objects']
                    info = None
                    seen_seq[self.sequence] = 1 if self.sequence not in seen_seq.keys() else seen_seq[self.sequence] + 1
                    scr_id = seen_seq[self.sequence]
                    self.final_masks = np.zeros([self.num_frames, self.img_size[0], self.img_size[1]])
                    self.pad_info = utils_custom.apply_pad(self.final_masks[0])[1]
                    self.hpad1, self.wpad1 = self.pad_info[0][0], self.pad_info[1][0]
                    self.hpad2, self.wpad2 = self.pad_info[0][1], self.pad_info[1][1]
                    self.h_ds, self.w_ds = int((self.img_size[0] + sum(self.pad_info[0])) / 4), int((self.img_size[1] + sum(self.pad_info[1])) / 4)
                    self.anno_6chEnc_r4_list = []
                    self.anno_3chEnc_r4_list = []
                    self.prob_map_of_frames = torch.zeros((self.num_frames, self.n_objects + 1, 4 * self.h_ds, 4 * self.w_ds)).cuda()
                    self.gt_masks = self.Davisclass.load_annotations(self.sequence)
                    self.scores_ni_nf = np.zeros([8, self.num_frames])

                    IoU_over_eobj = []

                else:
                    n_interaction += 1

                self.save_logger.printNlog('\nRunning sequence {} in (scribble index: {}) (round: {})'
                                           .format(self.sequence, sess.samples[sess.sample_idx][1], n_interaction))

                annotated_now = interactive_utils.scribbles.annotated_frames(sess.sample_last_scribble)[0]
                anno_dict['frames'].append(annotated_now)  # Where we save annotated frames
                anno_dict['masks_tobe_modified'].append(self.final_masks[annotated_now])  # mask before modefied at the annotated frame

                # Get Predicted mask & Mask decision from pred_mask
                self.final_masks = self.run_VOS_singleiact(n_interaction, scribbles, anno_dict['frames'])  # self.final_mask changes

                if self.config.test_save_pngs_option:
                    utils_custom.mkdir(
                        os.path.join(self.save_res_dir, 'result_video', '{}-scr{:02d}/round{:02d}'.format(self.sequence, scr_id, n_interaction)))
                    for fr in range(self.num_frames):
                        savefname = os.path.join(self.save_res_dir, 'result_video',
                                                 '{}-scr{:02d}/round{:02d}'.format(self.sequence, scr_id, n_interaction),
                                                 '{:05d}.png'.format(fr))
                        tmpPIL = Image.fromarray(self.final_masks[fr].astype(np.uint8), 'P')
                        tmpPIL.putpalette(self._palette)
                        tmpPIL.save(savefname)

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
                sess.submit_masks(self.final_masks, next_scribble_frame_candidates)  # F, H, W

                # print sequence name
                if tmpseq != self.sequence:
                    tmpseq, numseq = self.sequence, numseq + 1
                print(str(numseq) + ':' + str(self.sequence) + '-' + str(seen_seq[self.sequence]) + '\n')

                ## Visualizers and Saver
                # IoU estimation
                jaccard = batched_jaccard(self.gt_masks,
                                          self.final_masks,
                                          average_over_objects=False,
                                          nb_objects=self.n_objects
                                          )  # frames, objid

                IoU_over_eobj.append(jaccard)

                # save final mask in anno_dict
                anno_dict['annotated_masks'].append(self.final_masks[annotated_now])  # mask after modefied at the annotated frame

                if self.max_nb_interactions == len(anno_dict['frames']):  # After Lastround -> total 90 iter
                    seq_scrid_name = self.sequence + str(scr_id)

                    # IoU manager
                    IoU_over_eobj = np.stack(IoU_over_eobj, axis=0)  # niact,frames,n_obj
                    IoUeveryround_perobj = np.mean(IoU_over_eobj, axis=1)  # niact,n_obj

                    # show scribble input and output
                    savefiledir = os.path.join(self.save_res_dir, 'debug_scribble')
                    utils_custom.mkdir(savefiledir)
                    savefilename = os.path.join(savefiledir, seq_scrid_name + '.jpg')
                    utils_visualize.visualize_scrib_interaction(scribbles, anno_dict, self.sequence, savefilename)

                    # plot IoU
                    if self.config.test_save_pngs_option:
                        savefiledir = os.path.join(self.save_res_dir, 'plot_IoU_perObj')
                        utils_custom.mkdir(savefiledir)
                        for obj_idx in range(self.n_objects):
                            savefilename = os.path.join(savefiledir, seq_scrid_name + '-obj' + str(obj_idx + 1) + '_first{:03d}final{:03d}.png'
                                                        .format(int(1000 * IoUeveryround_perobj[0, obj_idx]),
                                                                int(1000 * IoUeveryround_perobj[-1, obj_idx])))
                            utils_visualize.visualize_interactionIoU(IoU_over_eobj[:, :, obj_idx], seq_scrid_name + '-obj' + str(obj_idx + 1),
                                                                     anno_dict['frames'], save_dir=savefilename, show_propagated_region=True)

                    # write csv
                    for obj_idx in range(self.n_objects):
                        with open(self.save_csvsummary_dir, mode='a') as csv_file:
                            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            writer.writerow([self.sequence, str(obj_idx + 1), str(scr_id)] + list(IoUeveryround_perobj[:, obj_idx]))

        summary = sess.get_global_summary(save_file=self.save_res_dir + '/summary_' + sess.report_name[7:] + '.json')
        analyze_summary(self.save_res_dir + '/summary_' + sess.report_name[7:] + '.json', metric=metric)
        fps = self.n_operated_frames_accum / self.total_taken_times_accum
        self.save_logger.printNlog('n_operated_frames_accum={}'.format(str(self.n_operated_frames_accum)))
        self.save_logger.printNlog('total_taken_times_accum={}'.format(str(self.total_taken_times_accum)))
        self.save_logger.printNlog('fps={}'.format(str(fps)))

        # final_IOU = summary['curve'][metric][-1]
        average_IoU_per_round = summary['curve'][metric][1:-1]

        torch.cuda.empty_cache()
        model = None
        return average_IoU_per_round

    def run_VOS_singleiact(self, n_interaction, scribbles_data, annotated_frames):

        annotated_frames_np = np.array(annotated_frames)
        num_workers = 4
        annotated_now = annotated_frames[-1]
        scribbles_list = scribbles_data['scribbles']
        seq_name = scribbles_data['sequence']

        # output_masks = self.final_masks.copy().astype(np.float64)

        prop_list = utils_custom.get_prop_list(annotated_frames, annotated_now, self.num_frames, proportion=self.config.test_propagation_proportion)
        prop_fore = sorted(prop_list)[0]
        prop_rear = sorted(prop_list)[-1]

        # Interaction settings
        pm_ps_ns_3ch_t = []  # n_obj,3,h,w
        if n_interaction == 1:
            for obj_id in range(1, self.n_objects + 1):
                pos_scrimg = utils_custom.scribble_to_image(scribbles_list, annotated_now, obj_id,
                                                            dilation=self.config.scribble_dilation_param,
                                                            prev_mask=self.final_masks[annotated_now])
                pm_ps_ns_3ch_t.append(np.stack([np.ones_like(pos_scrimg) / 2, pos_scrimg, np.zeros_like(pos_scrimg)], axis=0))
            pm_ps_ns_3ch_t = np.stack(pm_ps_ns_3ch_t, axis=0)  # n_obj,3,h,w

        else:
            for obj_id in range(1, self.n_objects + 1):
                prev_round_input = (self.final_masks[annotated_now] == obj_id).astype(np.float32)  # H,W
                pos_scrimg, neg_scrimg = utils_custom.scribble_to_image(scribbles_list, annotated_now, obj_id,
                                                                        dilation=self.config.scribble_dilation_param,
                                                                        prev_mask=self.final_masks[annotated_now], blur=True,
                                                                        singleimg=False, seperate_pos_neg=True)
                pm_ps_ns_3ch_t.append(np.stack([prev_round_input, pos_scrimg, neg_scrimg], axis=0))
            pm_ps_ns_3ch_t = np.stack(pm_ps_ns_3ch_t, axis=0)  # n_obj,3,h,w
        pm_ps_ns_3ch_t = torch.from_numpy(pm_ps_ns_3ch_t).cuda()

        if (prop_list[0] != annotated_now) and (prop_list.count(annotated_now) != 2):
            print(str(prop_list))
            raise NotImplementedError
        print(str(prop_list))  # we made our proplist first backward, and then forward

        composed_transforms = transforms.Compose([tr.Normalize_ApplymeanvarImage(self.config.mean, self.config.var),
                                                  tr.ToTensor()])
        db_test = davis_2017.DAVIS2017(split='val', transform=composed_transforms, root=self.config.davis_dataset_dir,
                                               custom_frames=prop_list, seq_name=seq_name, rgb=True,
                                               obj_id=None, no_gt=True, retname=True, prev_round_masks=self.final_masks, )
        testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

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

            t_start = time.time()
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

                smallest_alpha = 0.5
                if flag == 1:
                    sorted_frames = annotated_frames_np[annotated_frames_np < annotated_now]
                    if len(sorted_frames) == 0:
                        alpha = 1
                    else:
                        closest_addianno_frame = np.max(sorted_frames)
                        alpha = smallest_alpha + (1 - smallest_alpha) * (
                                (operating_frame - closest_addianno_frame) / (annotated_now - closest_addianno_frame))
                else:
                    sorted_frames = annotated_frames_np[annotated_frames_np > annotated_now]
                    if len(sorted_frames) == 0:
                        alpha = 1
                    else:
                        closest_addianno_frame = np.min(sorted_frames)
                        alpha = smallest_alpha + (1 - smallest_alpha) * (
                                (closest_addianno_frame - operating_frame) / (closest_addianno_frame - annotated_now))

                one_hot_outputs_t = (alpha * one_hot_outputs_t) + ((1 - alpha) * self.prob_map_of_frames[operating_frame])
                targ_onehot_prob = one_hot_outputs_t.clone()[1:].unsqueeze(1) # [nobj, 1, P_H, P_W]

            # Final mask indexing
            self.prob_map_of_frames[operating_frame] = one_hot_outputs_t
            self.scores_ni_nf[n_interaction-1,operating_frame] = score
            self.n_operated_frames_accum += 1
            self.total_taken_times_accum += time.time()-t_start

        output_masks = torch.argmax(self.prob_map_of_frames,dim=1).cpu().numpy().astype(np.uint8)[:,self.hpad1:-self.hpad2, self.wpad1:-self.wpad2]

        torch.cuda.empty_cache()

        return output_masks


    def soft_aggregation(self, ps):
        num_objects, H, W = ps.shape
        em = torch.zeros(num_objects +1, H, W).cuda()
        em[0] =  torch.prod(1-ps, dim=0) # bg prob
        em[1:num_objects+1] = ps # obj prob
        em = torch.clamp(em, 1e-7, 1-1e-7)
        logit = torch.log((em /(1-em)))
        return logit



if __name__ == '__main__':
    config = Config()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.test_gpu_id)

    tester = Main_tester(config)
    tester.run_for_diverse_metrics()
