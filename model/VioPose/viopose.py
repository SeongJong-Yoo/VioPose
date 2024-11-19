import tensorflow as tf
from tensorflow import keras
from ..model_base import *
from einops.layers.tensorflow import Rearrange
from utility.utils import read_model
import math

class Model(tf.keras.Model):
    def __init__(self,
                 pose_module,
                 dynamic_module,
                 audio,
                 audio_type,
                 hidden_dim_ratio=2,
                 num_keypoints=14,
                 num_heads=8,
                 dropout=0.1,
                 output_frame=300,
                 **kwargs):
        super(Model, self).__init__( **kwargs)
        
        self.pose_module_layers = pose_module.LAYERS
        self.pose_module_cross_layers = pose_module.CROSS_LAYERS
        self.pose_module_type = pose_module.TYPE
        self.pose_module_dim = pose_module.DIM
        self.input_frame = pose_module.FRAME

        self.dynamic_module_layers = 3
        self.dynamic_module_dim = dynamic_module.DIM
        self.dynamic_module_dim_ratio = dynamic_module.HIDDEN_DIM_RATIO

        self.audio_type = audio_type
        self.representation_type = audio.REPRESENTATION_TYPE
        self.audio_frame = audio.FRAME
        self.cnn_type = audio.CNN_TYPE
        self.cnn_filters = audio.CNN_FILTERS
        self.cnn_kernels = audio.CNN_KERNELS
        self.cnn_strides = audio.CNN_STRIDES
        self.trans_layers = audio.TRANS_LAYERS
        self.audio_cross_layers = audio.CROSS_LAYERS
        self.audio_dim = audio.DIM
        self.cnn_layers = 0
        self.audio_module = True

        if self.trans_layers==0:
            self.audio_module = False

        self.dropout_ratio = dropout
        self.num_keypoints = num_keypoints
        self.output_frame = output_frame
        self.hidden_dim_ratio = hidden_dim_ratio
        self.num_heads = num_heads

        # self.upsampling_frame = int(100 * self.output_frame / 30)
        
        if len(self.cnn_filters) > 0:
            self.cnn_layers = len(self.cnn_filters)
        self.audio_frame_discount = 1
        if self.cnn_layers > 0:
            for v in self.cnn_strides:
                self.audio_frame_discount *= v

        # Pose Module Blocks
        if self.pose_module_layers > 0:
            self.pose_module_embedding = LinearEmbedding(self.pose_module_dim, name="upsampling_embedding")

            if self.pose_module_type=='spatial':
                self.pose_module_PE = PositionEmbedding(num_keypoints, self.pose_module_dim, name="upsampling_PE")
                if not (num_keypoints * self.pose_module_dim) == self.dynamic_module_dim:
                    self.pose_module_to_temporal = tf.keras.layers.Dense(units=self.dynamic_module_dim, kernel_initializer=create_initializer(0.02), name="upsampling_to_temporal")
            elif self.pose_module_type=='temporal':
                self.pose_module_PE = PositionEmbedding(self.input_frame, self.pose_module_dim, name="upsampling_PE")
                if not self.pose_module_dim==self.dynamic_module_dim:
                    self.upsampling_to_temporal = tf.keras.layers.Dense(units=self.dynamic_module_dim, kernel_initializer=create_initializer(0.02), name="upsampling_to_temporal")

            self.pose_module_block = [
                Transformer(dim=self.pose_module_dim,
                            hidden_dim=hidden_dim_ratio*self.pose_module_dim,
                            dropout=dropout,
                            num_heads=num_heads,
                            att_dropout=dropout,
                            att_bias=False,
                            name=f"upsampling_block_{i+1}")
                            for i in range(self.pose_module_layers)]
            self.pose_module_cross_block = [
                CrossTransformer(dim=self.dynamic_module_dim,
                            hidden_dim=hidden_dim_ratio * self.dynamic_module_dim,
                            dropout=dropout,
                            num_heads=num_heads,
                            att_dropout=dropout,
                            att_bias=False,
                            name=f"upsampling_cross_block_{i+1}")
                            for i in range(self.pose_module_cross_layers)]
        if self.pose_module_cross_layers > 0:
            self.pose_module_cross_norm =  tf.keras.layers.LayerNormalization(epsilon=1e-5, name="up_cross_norm")
            self.pose_module_cross_PE = PositionEmbedding(int(self.audio_frame/self.audio_frame_discount), self.audio_dim, name="upsampling_cross_PE")

        self.to_full_frame = tf.keras.layers.Dense(units=3*num_keypoints, kernel_initializer=create_initializer(0.02), name="to_full")
        self.vel_regression_mlp = tf.keras.layers.Dense(units=3*num_keypoints, kernel_initializer=create_initializer(0.02), name="Vel_regression")

        dim = self.dynamic_module_dim
        if not self.audio_module:
            dim = self.dynamic_module_dim
        
        # Vel Net
        self.vel_PEs = PositionEmbedding(self.output_frame, dim, name=f"Vel_PE")
        self.vel_blocks = [
            Transformer(dim=dim,
                            hidden_dim=dim * self.hidden_dim_ratio,
                            num_heads=self.num_heads,
                            dropout=self.dropout_ratio,
                            name=f"Vel_block_{i+1}")
                            for i in range(self.dynamic_module_layers)]
        self.vel_norm = [tf.keras.layers.BatchNormalization(momentum=0.9, name=f"Vel_norm_{i+1}")
                         for i in range(self.dynamic_module_layers)]

        # Pose Net
        self.pose_PEs = PositionEmbedding(self.output_frame, dim, name=f"Pose_PE")
        self.pose_blocks = [
            Transformer(dim=dim,
                            hidden_dim=dim * self.hidden_dim_ratio,
                            num_heads=self.num_heads,
                            dropout=self.dropout_ratio,
                            name=f"Pose_block_{i+1}")
                            for i in range(self.dynamic_module_layers)]
        self.pose_norm = [tf.keras.layers.BatchNormalization(momentum=0.9, name=f"Pose_norm_{i+1}")
                          for i in range(self.dynamic_module_layers)]

        # Audio Net
        if self.cnn_layers > 0:
            if self.cnn_type=='1D':
                self.audio_cnn_block = [
                keras.layers.Conv1D(self.cnn_filters[i],
                                    self.cnn_kernels[i],
                                    self.cnn_strides[i],
                                    padding='same',
                                    activation='relu',
                                    name=f"Audio_CNN_1D_{i+1}")
                                    for i in range(self.cnn_layers)]
                
                self.num_maxpool = self.cnn_layers
                if self.audio_type=='features':
                    self.num_maxpool = 3
                        
            elif self.cnn_type=='2D':
                self.audio_cnn_block = [
                    keras.layers.Conv2D(self.cnn_filters[i],
                                        self.cnn_kernels[i],
                                        self.cnn_strides[i],
                                        padding='same',
                                        activation='relu',
                                        name=f"Audio_CNN_2D_{i+1}")
                                        for i in range(self.cnn_layers)]
                
            self.audio_batch_norm_block = [
                keras.layers.BatchNormalization(momentum=0.9, name=f"Audio_batch_norm_{i+1}")
                for i in range(self.cnn_layers)]
            self.fuse_batch_norm_block = [tf.keras.layers.BatchNormalization(momentum=0.9, name=f"fuse_batch_norm_{i+1}")
                    for i in range(self.dynamic_module_layers)]
            
        if self.trans_layers > 0:
            if self.audio_cross_layers > 0:
                self.fusion_PE = PositionEmbedding(self.output_frame, self.dynamic_module_dim, name="fusion_PE")
            self.fusion_embedding = tf.keras.layers.Dense(units=self.dynamic_module_dim, kernel_initializer=create_initializer(0.02), name="fusion_embedding")
            self.audio_embedding = LinearEmbedding(self.audio_dim, name="Audio_audio_embedding")
            self.audio_PE = PositionEmbedding(self.output_frame, self.audio_dim, name="Audio_audio_PE")
            self.audio_trans_block = [
                Transformer(dim=self.audio_dim,
                            hidden_dim=self.audio_dim * self.hidden_dim_ratio,
                            num_heads=self.num_heads,
                            dropout=self.dropout_ratio,
                            name=f"Audio_audio_transformer_{i+1}")
                            for i in range(self.trans_layers)]
            self.fusion_block = [
                Transformer(dim=self.dynamic_module_dim,
                            hidden_dim=self.audio_dim * self.hidden_dim_ratio,
                            num_heads=self.num_heads,
                            dropout=self.dropout_ratio,
                            name=f"Audio_cross_transformer_{i+1}")
                            for i in range(self.audio_cross_layers)]
            self.audio_resample_mlp = tf.keras.layers.Dense(units=self.input_frame, kernel_initializer=create_initializer(0.02), name="Audio_resample")
                
        else:
            self.audio_PE = PositionEmbedding(self.output_frame, self.dynamic_module_dim, name="Audio_audio_PE")
            self.fusion_block = [
                Transformer(dim=self.dynamic_module_dim,
                            hidden_dim=self.dynamic_module_dim * self.hidden_dim_ratio,
                            num_heads=self.num_heads,
                            dropout=self.dropout_ratio,
                            name=f"Audio_cross_transformer_{i+1}")
                            for i in range(self.pose_module_layers)]

        self.audio_last_mlp2 = tf.keras.layers.Dense(units=128, activation=keras.layers.LeakyReLU(), kernel_initializer=create_initializer(0.02), name="Audio_last_mlp2")
        self.audio_last_mlp3 = tf.keras.layers.Dense(units=64, activation=keras.layers.LeakyReLU(), kernel_initializer=create_initializer(0.02), name="Audio_last_mlp3")
        self.audio_regression = tf.keras.layers.Dense(units=3*self.num_keypoints, kernel_initializer=create_initializer(0.02), name="Audio_to_rotation")

    def data_prep(self, input):
        x_joint = input[0]
        if self.audio_type=='raw':
            x_audio = input[1][:, :, tf.newaxis]
        elif self.cnn_type=='2D':
            if input[1].shape[-1]==36:
                x_audio = input[1][:, :, :-1, tf.newaxis] # (batch, frame, dim, 1) for 2D CNN
            else:
                x_audio = input[1][:, :, :, tf.newaxis]
        else:
            if input[1].shape[-1]==36:
                x_audio = input[1][:, :, :-1]
            else:
                x_audio = input[1]                   # (batch, frame, dim)

        x_joint = Rearrange("b f j c-> b f (j c)")(x_joint)
        if self.pose_module_layers > 0:
            if self.pose_module_type=='spatial':
                x_joint = Rearrange("b f (j c)-> (b f) j c", c=2)(x_joint)

        return x_joint, x_audio

    def pose_module_construct(self, x, training=False):
        x = self.pose_module_embedding(x)
        x = self.pose_module_PE(x)

        att_weight = []
        for block in self.pose_module_block:
            x, att = block(x=x, training=training)
            att_weight.append(att)

        if self.pose_module_type=='spatial':
            x = Rearrange("(b f) j c -> b f (j c)", f=self.input_frame)(x)
            if not (self.num_keypoints * self.pose_module_dim) == self.dynamic_module_dim:
                x = self.pose_module_to_temporal(x) # (batch, output_frames, temp_dim)
        else:
            if not self.pose_module_dim==self.dynamic_module_dim:
                x = self.pose_module_to_temporal(x) # (batch, output_frames, temp_dim)

        return x, att_weight

    def full_frame_regression(self, x, vel):
        full_frame = self.to_full_frame(x) # (batch, frames, 3 * joints)

        pose = Rearrange("b f (j c)-> b f j c", c=3)(full_frame)

        pred_pose = pose[:, :-1, :] + vel[:, :-1, :]
        pred_pose = tf.concat((pred_pose[:, 0:1, :], pred_pose), axis=1)
        pose = (pose + pred_pose) / 2

        return pose

    def construct_cnn1d(self, x, training):
        for idx, cnn in enumerate(self.audio_cnn_block):
            x = cnn(x)
            x = self.audio_batch_norm_block[idx](x, training=training)
            idx_maxpool = self.num_maxpool - self.cnn_layers + idx
        return x

    def construct_cnn2d(self, x, training):
        for idx, cnn in enumerate(self.audio_cnn_block):
            x = cnn(x)
            x = self.audio_batch_norm_block[idx](x, training=training)
        x = Rearrange('b t d f->b t (d f)')(x)
        return x

    def audio_net(self, audio, x_joint, training=False):
        if self.cnn_layers > 0:
            if self.cnn_type=='1D':
                audio = self.construct_cnn1d(audio, training)
            elif self.cnn_type=='2D':
                audio = self.construct_cnn2d(audio, training)
                
        audio = tf.transpose(audio, perm=[0, 2, 1])
        audio = self.audio_resample_mlp(audio)
        audio = tf.transpose(audio, perm=[0, 2, 1])
        audio_att = []
        audio_feature=[]
        audio_cross_att = []
        fusion_lists = []
        if self.trans_layers > 0:
            audio = self.audio_embedding(audio)
            audio = self.audio_PE(audio)
            for block in self.audio_trans_block:
                audio, att = block(x=audio, training=training)
                audio_att.append(att)
                audio_feature.append(audio)

            fusion = tf.concat((audio, x_joint), axis=-1)
            fusion = self.fusion_embedding(fusion)
            fusion = self.fusion_PE(fusion)
            mid_fusion = fusion
            for i, block in enumerate(self.fusion_block):
                fusion, att = block(x=fusion, training=training)
                audio_cross_att.append(att)
                fusion = self.fuse_batch_norm_block[i](fusion + audio_feature[i], training=training)
                fusion_lists.append(fusion)

        att_weight = {"self": audio_att, "cross": audio_cross_att}
        return fusion_lists, mid_fusion, att_weight
    
    def tmp_net(self, x_joint, training=False):      
        fusion_lists = []
        x_joint = self.audio_PE(x_joint)

        mid_fusion = x_joint
        for block in self.fusion_block:
            x_joint, att = block(x=x_joint, training=training)
            fusion_lists.append(x_joint)

        return fusion_lists, mid_fusion
    
    def audio_to_pose(self, audio):
        x = self.audio_last_mlp2(audio)
        x = self.audio_last_mlp3(x)
        audio = self.audio_regression(x)

        audio_pose = Rearrange("b f (j c)->b f j c", j=self.num_keypoints)(audio)

        return audio_pose

    def vel_regression(self, latent, acc):
        vel = self.vel_regression_mlp(latent)

        vel = Rearrange("b f (j c)->b f j c", j=self.num_keypoints)(vel)

        vel_pred = vel[:, :-1, :] + acc[:, :-1, :]
        vel_pred = tf.concat((vel_pred[:, 0:1 :], vel_pred), axis=1)
        vel = (vel + vel_pred) / 2

        return vel
    
    def vel_net(self, fusion_lists, mid_fusion, training=False):
        vel_lists = []
        latent = mid_fusion
        latent = self.vel_PEs(latent)
        for i, block in enumerate(self.vel_blocks):
            latent, att = block(x=latent, training=training)
            latent = self.vel_norm[i](latent + fusion_lists[i])
            vel_lists.append(latent)

        return vel_lists
    
    def pose_net(self, vel_lists, mid_fusion, training=False):
        latent = mid_fusion
        latent = self.pose_PEs(latent)
        for i, block in enumerate(self.pose_blocks):
            latent, att = block(x=latent, training=training)
            latent = self.pose_norm[i](latent + vel_lists[i])

        return latent

    @tf.function
    def call(self, input, training=False):
        joints_list = []
        att_spatial = {}
        x_joint_init = input[0]
        joints_list.append(x_joint_init)
        x_joint, x_audio = self.data_prep(input)
        x_joint, att_spatial['self'] = self.pose_module_construct(x=x_joint, training=training)  # batch, high frame, dim

        if self.audio_module:  
            # Audio Module
            fusion_lists, mid_fusion, att_audio = self.audio_net(audio=x_audio, x_joint=x_joint, training=training)
        else:
            fusion_lists, mid_fusion = self.tmp_net(x_joint=x_joint, training=training)


        vel_lists = self.vel_net(fusion_lists, mid_fusion, training=training)

        pose_latent = self.pose_net(vel_lists, mid_fusion, training=training)

        acc = self.audio_to_pose(audio=fusion_lists[-1])
        vel = self.vel_regression(vel_lists[-1], acc)
        joints_dws = self.full_frame_regression(pose_latent, vel)
        joints_list.append(joints_dws)

        ## Final bottom-up refinement
        comp_vel = joints_dws[:, 1:] - joints_dws[:, :-1]
        comp_vel = tf.concat((comp_vel, comp_vel[:, -1:, :]), axis=1)
        vel = (vel + comp_vel)/2

        comp_acc = vel[:, 1:] - vel[:, :-1]
        comp_acc = tf.concat((comp_acc, comp_acc[:, -1:, :]), axis=1)
        acc = (acc + comp_acc)/2

        output = {"spatial": joints_list}
        output['acc'] = acc
        output['vel'] = vel

        return output


def construct_model(cfg, audio_type, pos_x_dim, audio_x_dim, print_summary=True):
    print(cfg.NUM_HEADS)
    fusion_net = Model(hidden_dim_ratio=cfg.HIDDEN_DIM_RATIO,
                       num_keypoints=cfg.NUM_KEYPOINTS,
                       audio_type=audio_type,
                       pose_module=cfg.POSE_MODULE,
                       dynamic_module=cfg.DYNAMIC_MODULE,
                       audio=cfg.AUDIO_MODULE,
                       num_heads=cfg.NUM_HEADS,
                       dropout=cfg.DROPOUT,
                       output_frame=cfg.OUTPUT_FRAME)
    
    fusion_net.build(input_shape=[pos_x_dim, audio_x_dim])

    if print_summary:
        fusion_net.summary()
    return fusion_net