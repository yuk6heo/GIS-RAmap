import os

class Config(object):
    def __init__(self):

        ################################ Common Variable parameters ##################################
        # DAVIS path
        self.davis_dataset_dir = '/home/yuk/data_ssd/datasets/DAVIS'
        self.youtube_dataset_dir = '/home/yuk/data_ssd/datasets/YoutubeVOS2019'
        self.test_gpu_id = 2
        self.test_metric_list = ['J', 'J_AND_F']  # one of ['J'], ['J_AND_F'], ['J', 'J_AND_F']
        self.test_guide_method = 'wo_RS'  # one of 'wo_RS', 'RS1', 'RS4'
        self.test_save_pngs_option = False

        self.result_dir_custom = 'results'

        ################################ For test parameters ##################################
        self.test_host = 'localhost'  # 'localhost' for subsets train and val.
        self.test_subset = 'val'
        self.test_userkey = None
        self.test_propagation_proportion = 0.99
        self.test_propth = 0.8
        self.test_min_nb_nodes = 2
        self.test_n_workers = 20

        ################################ For youtube test ##################################


        ############################### Other parameters ##################################
        self.mean, self.var = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.scribble_dilation_param = 5

        # Rel path
        # project_path = os.path.dirname(__file__)
        self.font_dir = 'etc/fonts/'
        self.palette_dir = self.davis_dataset_dir + '/Annotations/480p/bear/00000.png'
        self.test_result_df_dir = self.result_dir_custom + '/test_result_davisframework'
        self.test_result_yt_dir = self.result_dir_custom + '/test_result_youtube'
        self.test_result_rw_dir = self.result_dir_custom + '/test_result_realworld'
        self.test_load_state_dir = 'checkpoints/GIS-ckpt_standard.pth'  # CKpath
        self.test_load_state_youtubeval_dir = 'checkpoints/GIS-ckpt_wo_YTIVOStrainset.pth'  # CKpath