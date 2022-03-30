import logging

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import cv2
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from nilt_base.NILTlogger import get_logger

logger = get_logger("NILTlogger.morhping")


class MorphModel(tf.keras.Model):
    def __init__(self, map_size=96):
        super(MorphModel, self).__init__()
        self.map_size = map_size
        self.conv1 = tf.keras.layers.Conv2D(64, (5, 5))
        self.act1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.conv2 = tf.keras.layers.Conv2D(64, (5, 5))
        self.act2 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.convo = tf.keras.layers.Conv2D((3 + 3 + 2) * 2, (5, 5))

    def call(self, maps):
        x = tf.image.resize(maps, [self.map_size, self.map_size])
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.convo(x)
        return x


class Morph:
    # Default settings
    train_epochs = 1000

    im_sz = 1024
    mp_sz = 96

    warp_scale = 0.05
    mult_scale = 0.4
    add_scale = 0.4
    add_first = False

    preds = None
    save_preds = False
    plot_loss = True
    origins = None
    targets = None
    fps = 30

    def __init__(self, output_folder="", **kwargs):

        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        for kw, val in kwargs.items():
            assert hasattr(self, kw), f"No attribute to set with name '{kw}'"
            setattr(self, kw, val)
            logger.debug(f"Setting {kw}={val}")

    def load_image_file(self, image):
        """ Load image from file.

        Image will be resized fit the image shape needed by the network.

        Parameters
        ----------
        image : str
            Path to image to load

        Returns
        -------
        np.ndarray
            Image as a numpy array
        """
        dom = cv2.imread(image, cv2.IMREAD_COLOR)
        dom = cv2.cvtColor(dom, cv2.COLOR_BGR2RGB)

        if not dom.shape[0] == dom.shape[1]:
            logger.debug(f"Image ({image}) not square. Shape: {dom.shape}")

        if not dom.shape[0] == dom.shape[1] == self.im_sz:
            logger.debug(f"Dimensions of image ({image}) is not equal to the set image size ({self.im_sz}). Resizing.")
            dom = cv2.resize(dom, (self.im_sz, self.im_sz), interpolation=cv2.INTER_AREA)

        dom = dom / 127.5 - 1
        dom = dom.reshape(1, self.im_sz, self.im_sz, 3).astype(np.float32)

        return dom

    @tf.function
    def warp(self, origins, targets, preds_org, preds_trg):
        if self.add_first:
            scaling = tf.maximum(0.1, 1 + preds_org[:, :, :, 0:3] * self.mult_scale)
            res_targets = tfa.image.dense_image_warp(
                (origins + preds_org[:, :, :, 3:6] * 2 * self.add_scale) * scaling,
                preds_org[:, :, :, 6:8] * self.im_sz * self.warp_scale)

            scaling = tf.maximum(0.1, 1 + preds_trg[:, :, :, 0:3] * self.mult_scale)
            res_origins = tfa.image.dense_image_warp(
                (targets + preds_trg[:, :, :, 3:6] * 2 * self.add_scale) * scaling,
                preds_trg[:, :, :, 6:8] * self.im_sz * self.warp_scale)
        else:
            res_targets = tfa.image.dense_image_warp(
                origins * tf.maximum(0.1, 1 + preds_org[:, :, :, 0:3] * self.mult_scale) + preds_org[:, :, :,
                                                                                           3:6] * 2 * self.add_scale,
                preds_org[:, :, :, 6:8] * self.im_sz * self.warp_scale)
            res_origins = tfa.image.dense_image_warp(
                targets * tf.maximum(0.1, 1 + preds_trg[:, :, :, 0:3] * self.mult_scale) + preds_trg[:, :, :,
                                                                                           3:6] * 2 * self.add_scale,
                preds_trg[:, :, :, 6:8] * self.im_sz * self.warp_scale)

        return res_targets, res_origins

    def create_grid(self, scale):
        grid = np.mgrid[0:scale, 0:scale] / (scale - 1) * 2 - 1
        grid = np.swapaxes(grid, 0, 2)
        grid = np.expand_dims(grid, axis=0)
        return grid

    def load_preds(self, path):
        self.preds = np.load(path)

    def produce_warp_maps(self, origins, targets):

        self.origins = origins
        self.targets = targets

        model = MorphModel(map_size=self.mp_sz)

        loss_object = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

        train_loss = tf.keras.metrics.Mean(name='train_loss')

        @tf.function
        def train_step(maps, origins, targets):
            with tf.GradientTape() as tape:
                preds = model(maps)
                preds = tf.image.resize(preds, [self.im_sz, self.im_sz])

                # a = tf.random.uniform([maps.shape[0]])
                # res_targets, res_origins = self.warp(origins, targets, preds[...,:8] * a, preds[...,8:] * (1 - a))
                res_targets_, res_origins_ = self.warp(origins, targets, preds[..., :8], preds[..., 8:])

                res_map = tfa.image.dense_image_warp(maps, preds[:, :, :,
                                                           6:8] * self.im_sz * self.warp_scale)  # warp maps consistency checker
                res_map = tfa.image.dense_image_warp(res_map, preds[:, :, :, 14:16] * self.im_sz * self.warp_scale)

                loss = loss_object(maps, res_map) * 1 + loss_object(res_targets_, targets) * 0.3 + loss_object(
                    res_origins_,
                    origins) * 0.3

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss(loss)

        maps = self.create_grid(self.im_sz)
        maps = np.concatenate((maps, origins * 0.1, targets * 0.1), axis=-1).astype(np.float32)

        epoch = 0
        template = 'Epoch {}, Loss: {}'
        loss = np.full((self.train_epochs), fill_value=np.nan)

        for i in range(self.train_epochs):
            epoch = i + 1
            if epoch == 1:
                print(f"\tTraining ({self.train_epochs} epochs): 0%", end=" ")

            loss[i] = train_loss.result()

            train_step(maps, origins, targets)

            if (epoch < 100 and epoch % 10 == 0) or \
                    (epoch < 1000 and epoch % 100 == 0) or \
                    (epoch % 1000 == 0):

                # "Replace" tqdm
                print(f"{epoch/self.train_epochs*100:.0f}%", end=" ")

                preds = model(maps, training=False)[:1]
                preds = tf.image.resize(preds, [self.im_sz, self.im_sz])

                res_targets, res_origins = self.warp(origins, targets, preds[..., :8], preds[..., 8:])

                res_targets = tf.clip_by_value(res_targets, -1, 1)[0]
                res_img = ((res_targets.numpy() + 1) * 127.5).astype(np.uint8)
                cv2.imwrite(os.path.join(self.output_folder, "train/a_to_b_%d.jpg" % epoch),
                            cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))

                res_origins = tf.clip_by_value(res_origins, -1, 1)[0]
                res_img = ((res_origins.numpy() + 1) * 127.5).astype(np.uint8)
                cv2.imwrite(os.path.join(self.output_folder, "train/b_to_a_%d.jpg" % epoch),
                            cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))

                self.preds = preds.numpy()

        print("!! Done Training !!")

        if self.plot_loss:
            plt.figure()
            plt.semilogy(np.arange(1, self.train_epochs+1), loss, "*-")
            plt.title("Loss curve")
            plt.xlabel("Epoch")
            plt.xlabel("Loss")
            plt.savefig(os.path.join(self.output_folder, "loss_curve.png"))
            plt.close()

        if self.save_preds:
            np.save(os.path.join(self.output_folder, "preds.npy"), self.preds)

    def use_warp_maps(self, origins, targets, steps, fps=None):
        STEPS = steps

        if self.fps is None:
            fps = self.fps

        if self.preds is None:
            raise AttributeError(f"Weights are not defined, use produce_warp_maps() to train model or load_preds()"
                             f"to set weights from a file")
        preds = self.preds

        # save maps as images
        res_img = np.zeros((self.im_sz * 2, self.im_sz * 3, 3))

        res_img[self.im_sz * 0:self.im_sz * 1, self.im_sz * 0:self.im_sz * 1] = preds[0, :, :, 0:3]  # a_to_b add map
        res_img[self.im_sz * 0:self.im_sz * 1, self.im_sz * 1:self.im_sz * 2] = preds[0, :, :, 3:6]  # a_to_b mult map
        res_img[self.im_sz * 0:self.im_sz * 1, self.im_sz * 2:self.im_sz * 3, :2] = preds[0, :, :, 6:8]  # a_to_b warp map

        res_img[self.im_sz * 1:self.im_sz * 2, self.im_sz * 0:self.im_sz * 1] = preds[0, :, :, 8:11]  # b_to_a add map
        res_img[self.im_sz * 1:self.im_sz * 2, self.im_sz * 1:self.im_sz * 2] = preds[0, :, :, 11:14]  # b_to_a mult map
        res_img[self.im_sz * 1:self.im_sz * 2, self.im_sz * 2:self.im_sz * 3, :2] = preds[0, :, :, 14:16]  # b_to_a warp map

        res_img = np.clip(res_img, -1, 1)
        res_img = ((res_img + 1) * 127.5).astype(np.uint8)
        cv2.imwrite(os.path.join(self.output_folder, "maps.jpg"), cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))

        # apply maps and save results

        org_strength = tf.reshape(tf.range(STEPS, dtype=tf.float32), [STEPS, 1, 1, 1]) / (STEPS - 1)
        trg_strength = tf.reverse(org_strength, axis=[0])

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(os.path.join(self.output_folder, "morph.mp4"), fourcc, fps, (self.im_sz, self.im_sz))

        img_a = np.zeros((self.im_sz, self.im_sz * (STEPS // 10), 3), dtype=np.uint8)
        img_b = np.zeros((self.im_sz, self.im_sz * (STEPS // 10), 3), dtype=np.uint8)
        img_a_b = np.zeros((self.im_sz, self.im_sz * (STEPS // 10), 3), dtype=np.uint8)

        res_img = np.zeros((self.im_sz * 3, self.im_sz * (STEPS // 10), 3), dtype=np.uint8)

        all_im_path = os.path.join(self.output_folder, "all_steps")
        if not os.path.exists(all_im_path):
            os.mkdir(all_im_path)

        png_image_paths = []
        npy_image_paths = []

        for i in tqdm(range(STEPS), desc="Generating images"):
            preds_org = preds * org_strength[i]
            preds_trg = preds * trg_strength[i]

            res_targets, res_origins = self.warp(origins, targets, preds_org[..., :8], preds_trg[..., 8:])
            res_targets = tf.clip_by_value(res_targets, -1, 1)
            res_origins = tf.clip_by_value(res_origins, -1, 1)

            results = res_targets * trg_strength[i] + res_origins * org_strength[i]
            res_numpy = results.numpy()

            img = ((res_numpy[0] + 1) * 127.5).astype(np.uint8)
            video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            png_image_paths.append(os.path.join(all_im_path, f"step_{i:05d}.png"))
            npy_image_paths.append(os.path.join(all_im_path, f"step_{i:05d}.npy"))
            cv2.imwrite(png_image_paths[-1], cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            np.save(npy_image_paths[-1], img)

            if (i + 1) % 10 == 0:
                res_img[self.im_sz * 0:self.im_sz * 1, i // 10 * self.im_sz: (i // 10 + 1) * self.im_sz] = img
                res_img[self.im_sz * 1:self.im_sz * 2, i // 10 * self.im_sz: (i // 10 + 1) * self.im_sz] = (
                        (res_targets.numpy()[0] + 1) * 127.5).astype(np.uint8)
                res_img[self.im_sz * 2:self.im_sz * 3, i // 10 * self.im_sz: (i // 10 + 1) * self.im_sz] = (
                        (res_origins.numpy()[0] + 1) * 127.5).astype(np.uint8)

        cv2.imwrite(os.path.join(self.output_folder, "result.jpg"), cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))

        cv2.destroyAllWindows()
        video.release()
        logging.info(f"Result video saved to {os.path.join(self.output_folder, 'result.jpg')}.")

        return png_image_paths, npy_image_paths

    def generate_single_morphed(self, morph_pct, origins=None, targets=None):

        if origins is None:
            origins = self.origins
        if targets is None:
            targets = self.targets

        STEPS = 100

        if self.preds is None:
            raise AttributeError(f"Weights are not defined, use produce_warp_maps() to train model or load_preds()"
                             f"to set weights from a file")
        preds = self.preds

        if not 0 <= morph_pct <= 100:
            ValueError(f"Morph percentage should be an integer between 0-100. Got {morph_pct}")
        morph_pct = int(morph_pct)
        if morph_pct == 100:
            morph_pct = -1


        # apply maps and save results
        org_strength = tf.reshape(tf.range(STEPS, dtype=tf.float32), [STEPS, 1, 1, 1]) / (STEPS - 1)
        trg_strength = tf.reverse(org_strength, axis=[0])

        preds_org = preds * org_strength[morph_pct]
        preds_trg = preds * trg_strength[morph_pct]

        res_targets, res_origins = self.warp(origins, targets, preds_org[..., :8], preds_trg[..., 8:])
        res_targets = tf.clip_by_value(res_targets, -1, 1)
        res_origins = tf.clip_by_value(res_origins, -1, 1)

        results = res_targets * trg_strength[morph_pct] + res_origins * org_strength[morph_pct]
        res_numpy = results.numpy()

        img = ((res_numpy[0] + 1) * 127.5).astype(np.uint8)
        return img
