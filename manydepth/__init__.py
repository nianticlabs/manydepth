import os
import numpy as np
import time
import PIL.Image as pil
import torch
import matplotlib as mpl
import matplotlib.cm as cm

from .layers import transformation_from_parameters
from .test_simple import load_and_preprocess_intrinsics, preprocess_image
from .utils import download_model_if_doesnt_exist, manydepth_models_path, model_subfolder_names, git_dir
from . import networks

MODEL_NAMES = [
    "KITTI_MR_640_192",
    "KITTI_HR_1024_320",
    "CityScapes_512_192"
]

class manydepth:

    def __init__(self, model_name=MODEL_NAMES[0], no_cuda=False, intrinsics_json_path=os.path.join(git_dir, 'assets/test_sequence_intrinsics.json'), mode='multi') -> None:
        assert model_name in MODEL_NAMES, "Invalid Model Name"
        assert mode in ('multi', 'mono'), "Invalid Model Name"

        self.mode = mode
        
        if torch.cuda.is_available() and not no_cuda:
            self.device = torch.device("cuda")
            print("GPU Visible")
        else:
            self.device = torch.device("cpu")
            print("GPU not visible; CPU mode")
        
        # TODO download_model_if_doesnt_exist
        download_model_if_doesnt_exist(model_name=model_name)
        self.model_path = os.path.join(manydepth_models_path, model_name, model_subfolder_names[model_name])

        print("-> Loading model from ", self.model_path)

        # Loading pretrained model
        print("   Loading pretrained encoder")
        self.encoder_dict = torch.load(os.path.join(self.model_path, "encoder.pth"), map_location=self.device)
        self.encoder = networks.ResnetEncoderMatching(18, False,
                                                input_width=self.encoder_dict['width'],
                                                input_height=self.encoder_dict['height'],
                                                adaptive_bins=True,
                                                min_depth_bin=self.encoder_dict['min_depth_bin'],
                                                max_depth_bin=self.encoder_dict['max_depth_bin'],
                                                depth_binning='linear',
                                                num_depth_bins=96)

        filtered_dict_enc = {k: v for k, v in self.encoder_dict.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)

        print("   Loading pretrained decoder")
        self.depth_decoder = networks.DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        loaded_dict = torch.load(os.path.join(self.model_path, "depth.pth"), map_location=self.device)
        self.depth_decoder.load_state_dict(loaded_dict)

        print("   Loading pose network")
        self.pose_enc_dict = torch.load(os.path.join(self.model_path, "pose_encoder.pth"),
                                map_location=self.device)
        self.pose_dec_dict = torch.load(os.path.join(self.model_path, "pose.pth"), map_location=self.device)

        self.pose_enc = networks.ResnetEncoder(18, False, num_input_images=2)
        self.pose_dec = networks.PoseDecoder(self.pose_enc.num_ch_enc, num_input_features=1,
                                        num_frames_to_predict_for=2)

        self.pose_enc.load_state_dict(self.pose_enc_dict, strict=True)
        self.pose_dec.load_state_dict(self.pose_dec_dict, strict=True)

        # Setting states of networks
        self.encoder.eval()
        self.depth_decoder.eval()
        self.pose_enc.eval()
        self.pose_dec.eval()
        if torch.cuda.is_available():
            self.encoder.cuda()
            self.depth_decoder.cuda()
            self.pose_enc.cuda()
            self.pose_dec.cuda()

        
        self.K, self.invK = load_and_preprocess_intrinsics(intrinsics_json_path,
                                             resize_width=self.encoder_dict['width'],
                                             resize_height=self.encoder_dict['height'])

        pass

    def eval(self, input_image, source_image):
        """
            input_image  -> a test image to predict for
            source_image -> a previous image in the video sequence
        """

        input_image, original_size = preprocess_image(input_image, resize_width=self.encoder_dict['width'], resize_height=self.encoder_dict['height'])
        source_image, _ = preprocess_image(source_image, resize_width=self.encoder_dict['width'], resize_height=self.encoder_dict['height'])
        
        
        
        with torch.no_grad():

            # Estimate poses
            pose_inputs = [source_image, input_image]
            pose_inputs = [self.pose_enc(torch.cat(pose_inputs, 1))]
            axisangle, translation = self.pose_dec(pose_inputs)
            pose = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=True)

            if self.mode == 'mono':
                pose *= 0  # zero poses are a signal to the self.encoder not to construct a cost volume
                source_image *= 0

            # Estimate depth
            output, lowest_cost, _ = self.encoder(current_image=input_image,
                                            lookup_images=source_image.unsqueeze(1),
                                            poses=pose.unsqueeze(1),
                                            K=self.K,
                                            invK=self.invK,
                                            min_depth_bin=self.encoder_dict['min_depth_bin'],
                                            max_depth_bin=self.encoder_dict['max_depth_bin'])

            output = self.depth_decoder(output)

            sigmoid_output = output[("disp", 0)]
            sigmoid_output_resized = torch.nn.functional.interpolate(
                sigmoid_output, original_size, mode="bilinear", align_corners=False)
            sigmoid_output_resized = sigmoid_output_resized.cpu().numpy()[:, 0]

            # Saving numpy file
            #directory, filename = os.path.split(args.target_image_path)
            #output_name = os.path.splitext(filename)[0]
            #name_dest_npy = os.path.join(directory, "{}_disp_{}.npy".format(output_name, self.mode))
            #np.save(name_dest_npy, sigmoid_output.cpu().numpy())

            # Saving colormapped depth image and cost volume argmin
            for plot_name, toplot in (('costvol_min', lowest_cost), ('disp', sigmoid_output_resized)):
                toplot = toplot.squeeze()
                normalizer = mpl.colors.Normalize(vmin=toplot.min(), vmax=np.percentile(toplot, 95))
                mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                colormapped_im = (mapper.to_rgba(toplot)[:, :, :3] * 255).astype(np.uint8)
                im = pil.fromarray(colormapped_im)

                #name_dest_im = os.path.join(directory, "{}_{}_{}.jpeg".format(output_name, plot_name, self.mode))
                #im.save(name_dest_im)

                #print("-> Saved output image to {}".format(name_dest_im))

        return colormapped_im
        