from statsmodels.compat import scipy

from models import *
import keras
import scipy
from keras.losses import *
from sklearn.metrics import mean_squared_error
from keras.callbacks import *
from sklearn.preprocessing import StandardScaler
import gc
from scipy.stats import pearsonr
import numpy as np
import h5py
import random
from keras.utils import *
import matplotlib.pyplot as plt
from data_augmentation import *


class GetSCAMetric(Callback):
    def __init__(self, x_train, y_train, epochs, correct_key, loss, x_val, y_val):
        super().__init__()
        # parameters:
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.epochs = epochs
        self.correct_key = correct_key
        self.loss = loss

        """ 
        - loss value for each epoch and for each key candidate 
        - the loss is computed for the training and validation sets
        """
        self.loss_candidates_epochs_train = np.zeros((epochs, 256))
        self.loss_candidates_epochs_val = np.zeros((epochs, 256))

        """ 
        Correlation between true and predicted labels
        - correlation value for each epoch and for each key candidate 
        - the correlation is computed for the training and validation sets
        """
        self.corr_epochs_train = np.zeros((epochs, 256))
        self.corr_epochs_val = np.zeros((epochs, 256))

        """ 
        Key rank evolution for the correct key (we assume a known key analysis here to be able to compute key rank)
        Key rank is computed for training and validation sets, considering both loss and correlation metrics.
        """
        self.key_rank_evolution_loss_train = np.zeros(epochs)
        self.key_rank_evolution_corr_train = np.zeros(epochs)
        self.key_rank_evolution_loss_val = np.zeros(epochs)
        self.key_rank_evolution_corr_val = np.zeros(epochs)

        """
        Objective function to measure the quality of the model. The objective function value is computed for each training epoch. 
        """
        self.objective_function_from_loss_train = np.zeros(epochs)
        self.objective_function_from_corr_train = np.zeros(epochs)
        self.objective_function_from_loss_val = np.zeros(epochs)
        self.objective_function_from_corr_val = np.zeros(epochs)

        self.objective_function_from_corr_train = np.zeros(epochs)
        self.objective_function_from_corr_val = np.zeros(epochs)

    def softmax(self, x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=0)

    def compute_loss(self, y_pred, y_true):

        loss_candidates = np.zeros(256)
        if self.loss == "mse":
            for k in range(256):
                loss_candidates[k] = mean_squared_error(y_true[:, k], y_pred[:, k])

        if self.loss == "mae":
            for k in range(256):
                loss_candidates[k] = mean_absolute_error(y_true[:, k], y_pred[:, k])

        if self.loss == "huber":
            for k in range(256):
                delta = 1
                abs_error = np.mean(np.abs(y_true[:, k] - y_pred[:, k]))
                if abs_error <= delta:
                    loss = 0.5 * mean_squared_error(y_true[:, k], y_pred[:, k])
                else:
                    loss = np.mean(delta * (np.abs(y_true[:, k] - y_pred[:, k]) - 0.5 * delta))
                loss_candidates[k] = loss

        if self.loss == "corr":
            for k in range(256):
                loss_candidates[k] = 1 - abs(pearsonr(y_true[:, k], y_pred[:, k])[0])

        if self.loss == "key_rank":
            y_pred = 1 - (np.abs(y_true - y_pred) / 255)
            y_pred_k = [self.softmax(p) for p in y_pred]
            y_pred_k = np.log(np.array(y_pred_k) + 1e-36)

            for k in range(256):
                loss_candidates[k] = - np.mean(np.array(y_pred_k)[:, k])
            del y_pred_k

        if self.loss == "z_score_mse":
            for k in range(256):
                y_true_score = (y_true[:, k] - np.mean(y_true[:, k])) / np.maximum(float(np.std(y_true[:, k])), 1e-10)
                y_pred_score = (y_pred[:, k] - np.mean(y_pred[:, k])) / np.maximum(float(np.std(y_pred[:, k])), 1e-10)
                loss_candidates[k] = mean_squared_error(y_true_score, y_pred_score)

        if self.loss == "z_score_mae":
            for k in range(256):
                y_true_score = (y_true[:, k] - np.mean(y_true[:, k])) / np.maximum(float(np.std(y_true[:, k])), 1e-10)
                y_pred_score = (y_pred[:, k] - np.mean(y_pred[:, k])) / np.maximum(float(np.std(y_pred[:, k])), 1e-10)
                loss_candidates[k] = mean_absolute_error(y_true_score, y_pred_score)

        if self.loss == "z_score_huber":
            for k in range(256):
                y_true_score = (y_true[:, k] - np.mean(y_true[:, k])) / np.maximum(float(np.std(y_true[:, k])), 1e-10)
                y_pred_score = (y_pred[:, k] - np.mean(y_pred[:, k])) / np.maximum(float(np.std(y_pred[:, k])), 1e-10)
                delta = 1
                abs_error = np.mean(np.abs(y_true_score - y_pred_score))
                if abs_error <= delta:
                    loss = 0.5 * mean_squared_error(y_true_score, y_pred_score)
                else:
                    loss = np.mean(delta * (np.abs(y_true_score - y_pred_score) - 0.5 * delta))
                loss_candidates[k] = loss

        if self.loss == "z_score_corr":
            for k in range(256):
                y_true_score = (y_true[:, k] - np.mean(y_true[:, k])) / np.maximum(float(np.std(y_true[:, k])), 1e-10)
                y_pred_score = (y_pred[:, k] - np.mean(y_pred[:, k])) / np.maximum(float(np.std(y_pred[:, k])), 1e-10)
                loss_candidates[k] = 1 - abs(pearsonr(y_true_score, y_pred_score)[0])

        return loss_candidates

    def get_key_rank(self, epoch, loss_candidates_epochs, validation=False):
        k_sum_sorted = np.argsort(loss_candidates_epochs[epoch, :])
        found_key = k_sum_sorted[0]
        if validation:
            self.key_rank_evolution_loss_val[epoch] = list(k_sum_sorted).index(self.correct_key) + 1
            print(f"Key rank (val): {self.key_rank_evolution_loss_val[epoch]} (found_key: {found_key})")
        else:
            self.key_rank_evolution_loss_train[epoch] = list(k_sum_sorted).index(self.correct_key) + 1
            print(f"Key rank: {self.key_rank_evolution_loss_train[epoch]} (found_key: {found_key})")
        return found_key

    def get_key_rank_corr(self, y_pred, y_true, epoch, validation=False):
        corr = np.zeros(256)
        for key in range(256):
            corr[key] = abs(pearsonr(y_pred[:, key], y_true[:, key])[0])
        k_sum_sorted = np.argsort(corr)[::-1]
        found_key = k_sum_sorted[0]

        if validation:
            self.key_rank_evolution_corr_val[epoch] = list(k_sum_sorted).index(self.correct_key) + 1
            print(f"Key rank (val): {self.key_rank_evolution_corr_val[epoch]} (found_key: {found_key})")
        else:
            self.key_rank_evolution_corr_train[epoch] = list(k_sum_sorted).index(self.correct_key) + 1
            print(f"Key rank: {self.key_rank_evolution_corr_train[epoch]} (found_key: {found_key})")

        return found_key, corr

    def get_objective_function(self, epoch, found_key, loss_epochs):

        """
        Compute objective function from loss values between true and predicted labels
        This function subtracts loss value for the found key candidate from the average of wrong keys
        """

        loss_most_likely_key = loss_epochs[epoch, found_key]
        mean_loss_other_keys = 0
        for key in range(256):
            if key != found_key:
                mean_loss_other_keys += loss_epochs[epoch, key]
        mean_loss_other_keys /= 255
        return abs(loss_most_likely_key - mean_loss_other_keys)

    def on_epoch_end(self, epoch, logs=None):

        """ Predict training and validation sets """
        y_pred_train = self.model.predict(self.x_train)
        y_pred_val = self.model.predict(self.x_val)

        """ Compute loss for each key candidate """
        self.loss_candidates_epochs_train[epoch] = self.compute_loss(y_pred_train, self.y_train)
        self.loss_candidates_epochs_val[epoch] = self.compute_loss(y_pred_val, self.y_val)

        """ 
        Compute key rank for the correct key candidate 
        Key rank is computed from loss and correlation between true and predicted labels 
        """
        found_key_loss_train = self.get_key_rank(epoch, self.loss_candidates_epochs_train)
        found_key_corr_train, corr = self.get_key_rank_corr(y_pred_train, self.y_train, epoch)
        found_key_val = self.get_key_rank(epoch, self.loss_candidates_epochs_val, validation=True)
        found_key_corr_val, corr_val = self.get_key_rank_corr(y_pred_val, self.y_val, epoch, validation=True)

        self.objective_function_from_loss_train[epoch] = self.get_objective_function(epoch, found_key_loss_train,
                                                                                     self.loss_candidates_epochs_train)
        self.objective_function_from_corr_train[epoch] = self.get_objective_function(epoch, found_key_corr_train,
                                                                                     self.loss_candidates_epochs_train)
        self.objective_function_from_loss_val[epoch] = self.get_objective_function(epoch, found_key_val,
                                                                                   self.loss_candidates_epochs_val)
        self.objective_function_from_corr_val[epoch] = self.get_objective_function(epoch, found_key_corr_val,
                                                                                   self.loss_candidates_epochs_val)

        """ Take the maximum correlation values as the objective function"""
        self.objective_function_from_corr_train[epoch] = np.max(corr)
        self.objective_function_from_corr_val[epoch] = np.max(corr_val)
        self.corr_epochs_train[epoch] = corr
        self.corr_epochs_val[epoch] = corr_val

        del y_pred_train
        gc.collect()


def generate_random_hyperparameters(model_type="mlp"):
    """
    Function to generate a random set of hyperparameters
    """

    if model_type == "mlp":
        hp = {
            "neurons": random.choice([200, 300, 400, 500, 600, 700, 800, 900, 1000]),
            "batch_size": random.choice([50, 100]),
            "layers": random.choice([1, 2, 3, 4]),
            "activation": random.choice(["elu", "selu", "relu"]),
            "learning_rate": random.choice(
                [0.005, 0.0025, 0.002, 0.001, 0.0025, 0.0005, 0.0001, 0.0002, 0.00025, 0.00005]),
            "kernel_initializer": random.choice(
                ["random_uniform", "he_uniform", "glorot_uniform", "random_normal", "he_normal", "glorot_normal"]),
            # "kernel_regularizer": random.choice([None, "l1", "l2"]),
            "kernel_regularizer": random.choice([None]),
            # "dropout": random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5]),
            "dropout": random.choice([0]),
            "optimizer": random.choice(["Adam"])
        }
    else:
        hp = {
            "neurons": random.choice([200, 300, 400, 500, 600, 700, 800, 900, 1000]),
            "batch_size": random.choice([50, 100, 200, 300, 400, 500, 1000, 2000]),
            "layers": random.choice([1, 2]),
            "filters": random.choice([4, 8, 12, 16, 32]),
            "kernel_size": random.choice([5, 10, 20, 30, 40]),
            "pool_type": random.choice(["Average", "Max"]),
            "pool_size": random.choice([2, 4]),
            "conv_layers": random.choice([1, 2, 3, 4]),
            "activation": random.choice(["elu", "selu", "relu"]),
            "learning_rate": random.choice(
                [0.005, 0.0025, 0.002, 0.001, 0.0025, 0.0005, 0.0001, 0.0002, 0.00025, 0.00005]),
            "kernel_initializer": random.choice(
                ["random_uniform", "he_uniform", "glorot_uniform", "random_normal", "he_normal", "glorot_normal"]),
            # "kernel_regularizer": random.choice([None, "l1", "l2"]),
            "kernel_regularizer": random.choice([None]),
            # "dropout": random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5]),
            "dropout": random.choice([0]),
            "optimizer": random.choice(["Adam", "RMSprop"])
        }
        hp["pool_strides"] = hp["pool_size"]
        conv_stride_options = [1, 2, 3, 4, 5, 10, 15, 20]
        possible_stride_options = []
        for i, st in enumerate(conv_stride_options):
            if st <= hp["kernel_size"]:
                possible_stride_options.append(st)
        # hp["strides"] = random.choice(possible_stride_options)
        hp["strides"] = random.choice([1, 2, 3, 4])

    if hp["kernel_regularizer"] is not None:
        hp["kernel_regularizer_value"] = random.choice([0.0005, 0.0001, 0.0002, 0.00025, 0.00005, 0.00001])

    hp["seed"] = np.random.randint(1048576)

    return hp


aes_sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])

AES_Sbox_inv = np.array([
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
])

hamming = [bin(n).count("1") for n in range(256)]


class ReadAESRD:

    def __init__(self, n_attack, target_byte, leakage_model, file_path):
        self.n_attack = n_attack
        self.target_byte = target_byte
        self.leakage_model = leakage_model
        self.file_path = file_path
        self.classes = 9 if leakage_model == "HW" else 256

        self.attack_plaintexts = None
        self.attack_keys = None
        self.attack_masks = None

        self.x_attack = []
        self.y_attack = []
        self.attack_labels = []

        self.labels_key_hypothesis_attack = None
        self.share1_attack, self.share2_attack = None, None

        self.round_key = "00112233445566778899AABBCCDDEEFF"
        self.correct_key_attack = [43, 126, 21, 22, 40, 174, 210, 166, 171, 247, 21, 136, 9, 207, 79, 60][target_byte]

        self.load_dataset()

    def load_dataset(self):
        mat = scipy.io.loadmat(self.file_path)
        samples = mat['CompressedTraces']
        samples = np.transpose(samples)
        plaintexts = mat['plaintext']
        attack_key = [[43, 126, 21, 22, 40, 174, 210, 166, 171, 247, 21, 136, 9, 207, 79, 60] for _ in
                      range(len(samples))]

        self.attack_plaintexts = plaintexts[:self.n_attack]
        self.attack_keys = attack_key[:self.n_attack]

        self.x_attack = samples[:self.n_attack]
        self.attack_labels = self.aes_labelize(self.attack_plaintexts, self.attack_keys)
        self.y_attack = to_categorical(self.attack_labels, num_classes=self.classes)
        self.labels_key_hypothesis_attack = self.create_labels_key_guess(self.attack_plaintexts)
        """ uncomment to use data augmentation - either data_augmentation_gaussian_noise or data_augmentation_shifts can be used"""
        # if self.n_attack < 10000:
        #     augmented = self.n_attack
        # else:
        #     augmented = 10000
        # X2, Y2 = next(data_augmentation_gaussian_noise(self.x_attack, self.labels_key_hypothesis_attack.T, augmented, [len(self.x_attack[0]), 1]))
        #
        # self.x_attack = np.concatenate((self.x_attack, X2), axis=0)
        # self.labels_key_hypothesis_attack = np.concatenate((self.labels_key_hypothesis_attack, Y2.T), axis=1)

    def aes_labelize(self, plaintexts, keys):

        if np.array(keys).ndim == 1:
            """ repeat key if argument keys is a single key candidate (for GE and SR computations)"""
            keys = np.full([len(plaintexts), 16], keys)

        plaintext = [row[self.target_byte] for row in plaintexts]
        key = [row[self.target_byte] for row in keys]
        state = [int(p) ^ int(k) for p, k in zip(plaintext, key)]
        intermediates = aes_sbox[state]

        return [bin(iv).count("1") for iv in intermediates] if self.leakage_model == "HW" else intermediates

    def create_labels_key_guess(self, plaintexts):
        labels_key_hypothesis = np.zeros((256, len(plaintexts)), dtype='int64')
        for key_byte_hypothesis in range(256):
            key_h = bytearray.fromhex(self.round_key)
            key_h[self.target_byte] = key_byte_hypothesis
            labels_key_hypothesis[key_byte_hypothesis] = self.aes_labelize(plaintexts, key_h)
        return labels_key_hypothesis


class ReadAESHD:

    def __init__(self, n_attack, target_byte, leakage_model, file_path):
        self.n_attack = n_attack
        self.target_byte = target_byte
        self.leakage_model = leakage_model
        self.file_path = file_path
        self.classes = 9 if leakage_model == "HW" else 256

        self.attack_plaintexts = None
        self.attack_keys = None
        self.attack_masks = None

        self.x_attack = []
        self.y_attack = []
        self.attack_labels = []

        self.labels_key_hypothesis_attack = None
        self.share1_attack, self.share2_attack = None, None

        self.round_key = "00112233445566778899AABBCCDDEEFF"
        self.correct_key_attack = [208, 20, 249, 168, 201, 238, 37, 137, 225, 63, 12, 200, 182, 99, 12, 166][
            target_byte]

        self.load_dataset()

    def load_dataset(self):
        in_file = h5py.File(self.file_path, "r")

        attack_samples = np.array(in_file['Profiling_traces/traces'])
        attack_plaintext = in_file['Profiling_traces/metadata']['ciphertext']
        attack_key = in_file['Profiling_traces/metadata']['key']
        # attack_mask = in_file['Profiling_traces/metadata']['masks']

        self.attack_plaintexts = attack_plaintext[:self.n_attack]
        self.attack_keys = attack_key[:self.n_attack]
        # self.attack_masks = attack_mask[:self.n_attack]

        self.x_attack = attack_samples[:self.n_attack]
        self.attack_labels = self.aes_labelize(self.attack_plaintexts, self.attack_keys)
        self.y_attack = to_categorical(self.attack_labels, num_classes=self.classes)
        self.labels_key_hypothesis_attack = self.create_labels_key_guess(self.attack_plaintexts)

        """ uncomment to use data augmentation - either data_augmentation_gaussian_noise or data_augmentation_shifts can be used"""
        # if self.n_attack < 10000:
        #     augmented = self.n_attack
        # else:
        #     augmented = 10000
        # X2, Y2 = next(data_augmentation_gaussian_noise(self.x_attack, self.labels_key_hypothesis_attack.T, augmented, [len(self.x_attack[0]), 1]))
        #
        # self.x_attack = np.concatenate((self.x_attack, X2), axis=0)
        # self.labels_key_hypothesis_attack = np.concatenate((self.labels_key_hypothesis_attack, Y2.T), axis=1)

    def aes_labelize(self, plaintexts, keys):

        if np.array(keys).ndim == 1:
            """ repeat key if argument keys is a single key candidate (for GE and SR computations)"""
            keys = np.full([len(plaintexts), 16], keys)

        shift_row_mask = np.array([0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11])

        c_j = [cj[shift_row_mask[self.target_byte]] for cj in plaintexts]
        c_i = [ci[self.target_byte] for ci in plaintexts]
        """ get key byte from round key 10 """
        k_i = [ki[self.target_byte] for ki in keys]
        if self.leakage_model == 'HW':
            return [hamming[AES_Sbox_inv[int(ci) ^ int(ki)] ^ int(cj)] for ci, cj, ki in
                    zip(np.asarray(c_i[:]), np.asarray(c_j[:]), np.asarray(k_i[:]))]
        else:
            return [AES_Sbox_inv[int(ci) ^ int(ki)] ^ int(cj) for ci, cj, ki in
                    zip(np.asarray(c_i[:]), np.asarray(c_j[:]), np.asarray(k_i[:]))]

    def create_labels_key_guess(self, plaintexts):
        labels_key_hypothesis = np.zeros((256, len(plaintexts)), dtype='int64')
        for key_byte_hypothesis in range(256):
            key_h = bytearray.fromhex(self.round_key)
            key_h[self.target_byte] = key_byte_hypothesis
            labels_key_hypothesis[key_byte_hypothesis] = self.aes_labelize(plaintexts, key_h)
        return labels_key_hypothesis


class ReadASCADf:

    def __init__(self, n_attack, target_byte, leakage_model, file_path):
        self.n_attack = n_attack
        self.target_byte = target_byte
        self.leakage_model = leakage_model
        self.file_path = file_path
        self.classes = 9 if leakage_model == "HW" else 256

        self.attack_plaintexts = None
        self.attack_keys = None
        self.attack_masks = None

        self.x_attack = []
        self.y_attack = []
        self.attack_labels = []

        self.labels_key_hypothesis_attack = None
        self.share1_attack, self.share2_attack = None, None

        self.round_key = "00112233445566778899AABBCCDDEEFF"
        self.correct_key_attack = [77, 251, 224, 242, 114, 33, 254, 16, 167, 141, 74, 220, 142, 73, 4, 105][target_byte]

        self.load_dataset()

    def load_dataset(self):
        in_file = h5py.File(self.file_path, "r")

        attack_samples = np.array(in_file['Profiling_traces/traces'])
        attack_plaintext = in_file['Profiling_traces/metadata']['plaintext']
        attack_key = in_file['Profiling_traces/metadata']['key']
        attack_mask = in_file['Profiling_traces/metadata']['masks']

        self.attack_plaintexts = attack_plaintext[:self.n_attack]
        self.attack_keys = attack_key[:self.n_attack]
        self.attack_masks = attack_mask[:self.n_attack]

        self.x_attack = attack_samples[:self.n_attack]
        self.attack_labels = self.aes_labelize(self.attack_plaintexts, self.attack_keys)
        self.y_attack = to_categorical(self.attack_labels, num_classes=self.classes)
        self.labels_key_hypothesis_attack = self.create_labels_key_guess(self.attack_plaintexts)

        """ uncomment to use data augmentation - either data_augmentation_gaussian_noise or data_augmentation_shifts can be used"""
        # if self.n_attack < 10000:
        #     augmented = self.n_attack
        # else:
        #     augmented = 10000
        # X2, Y2 = next(data_augmentation_gaussian_noise(self.x_attack, self.labels_key_hypothesis_attack.T, augmented, [len(self.x_attack[0]), 1]))
        #
        # self.x_attack = np.concatenate((self.x_attack, X2), axis=0)
        # self.labels_key_hypothesis_attack = np.concatenate((self.labels_key_hypothesis_attack, Y2.T), axis=1)


    def aes_labelize(self, plaintexts, keys):

        if np.array(keys).ndim == 1:
            """ repeat key if argument keys is a single key candidate (for GE and SR computations)"""
            keys = np.full([len(plaintexts), 16], keys)

        plaintext = [row[self.target_byte] for row in plaintexts]
        key = [row[self.target_byte] for row in keys]
        state = [int(p) ^ int(k) for p, k in zip(plaintext, key)]
        intermediates = aes_sbox[state]

        return [bin(iv).count("1") for iv in intermediates] if self.leakage_model == "HW" else intermediates

    def create_labels_key_guess(self, plaintexts):
        labels_key_hypothesis = np.zeros((256, len(plaintexts)), dtype='int64')
        for key_byte_hypothesis in range(256):
            key_h = bytearray.fromhex(self.round_key)
            key_h[self.target_byte] = key_byte_hypothesis
            labels_key_hypothesis[key_byte_hypothesis] = self.aes_labelize(plaintexts, key_h)
        return labels_key_hypothesis


class ReadASCADr:

    def __init__(self, n_attack, target_byte, leakage_model, file_path):
        self.n_attack = n_attack
        self.target_byte = target_byte
        self.leakage_model = leakage_model
        self.file_path = file_path
        self.classes = 9 if leakage_model == "HW" else 256

        self.attack_plaintexts = None
        self.attack_keys = None
        self.attack_masks = None

        self.x_attack = []
        self.y_attack = []
        self.attack_labels = []

        self.labels_key_hypothesis_attack = None
        self.share1_attack, self.share2_attack = None, None

        self.round_key = "00112233445566778899AABBCCDDEEFF"
        self.correct_key_attack = bytearray.fromhex(self.round_key)[target_byte]

        self.load_dataset()

    def load_dataset(self):
        in_file = h5py.File(self.file_path, "r")

        attack_samples = np.array(in_file['Attack_traces/traces'])
        attack_plaintext = in_file['Attack_traces/metadata']['plaintext']
        attack_key = in_file['Attack_traces/metadata']['key']
        attack_mask = in_file['Attack_traces/metadata']['masks']

        self.attack_plaintexts = attack_plaintext[:self.n_attack]
        self.attack_keys = attack_key[:self.n_attack]
        self.attack_masks = attack_mask[:self.n_attack]

        self.x_attack = attack_samples[:self.n_attack]
        self.attack_labels = self.aes_labelize(self.attack_plaintexts, self.attack_keys)
        self.y_attack = to_categorical(self.attack_labels, num_classes=self.classes)
        self.labels_key_hypothesis_attack = self.create_labels_key_guess(self.attack_plaintexts)

        """ uncomment to use data augmentation - either data_augmentation_gaussian_noise or data_augmentation_shifts can be used"""
        # if self.n_attack < 10000:
        #     augmented = self.n_attack
        # else:
        #     augmented = 10000
        # X2, Y2 = next(data_augmentation_gaussian_noise(self.x_attack, self.labels_key_hypothesis_attack.T, augmented, [len(self.x_attack[0]), 1]))
        #
        # self.x_attack = np.concatenate((self.x_attack, X2), axis=0)
        # self.labels_key_hypothesis_attack = np.concatenate((self.labels_key_hypothesis_attack, Y2.T), axis=1)


    def aes_labelize(self, plaintexts, keys):

        if np.array(keys).ndim == 1:
            """ repeat key if argument keys is a single key candidate (for GE and SR computations)"""
            keys = np.full([len(plaintexts), 16], keys)

        plaintext = [row[self.target_byte] for row in plaintexts]
        key = [row[self.target_byte] for row in keys]
        state = [int(p) ^ int(k) for p, k in zip(plaintext, key)]
        intermediates = aes_sbox[state]

        return [bin(iv).count("1") for iv in intermediates] if self.leakage_model == "HW" else intermediates

    def create_labels_key_guess(self, plaintexts):
        labels_key_hypothesis = np.zeros((256, len(plaintexts)), dtype='int64')
        for key_byte_hypothesis in range(256):
            key_h = bytearray.fromhex(self.round_key)
            key_h[self.target_byte] = key_byte_hypothesis
            labels_key_hypothesis[key_byte_hypothesis] = self.aes_labelize(plaintexts, key_h)
        return labels_key_hypothesis


import sys


def multi_output_regression():
    """ Settings """
    model_type = "mlp"
    epochs = int(sys.argv[1])
    loss = "z_score_mse"

    """ Open dataset """
    dataset_path = "ASCAD/ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5"
    leakage_model = sys.argv[2]
    target_key_byte = 2
    n_attack = int(sys.argv[3])
    dataset = ReadASCADf(n_attack, target_key_byte, leakage_model, dataset_path)

    features = len(dataset.x_attack[0])

    """ Normalize attack set"""
    scaler = StandardScaler()
    dataset.x_attack = scaler.fit_transform(dataset.x_attack)
    dataset.x_attack = np.array(dataset.x_attack)

    """ Re-shape, if needed """
    if model_type == "cnn":
        dataset.x_attack = dataset.x_attack.reshape(dataset.x_attack.shape[0], dataset.x_attack.shape[1], 1)

    """ Get table of labels for all key candidates per trace """
    labels = dataset.labels_key_hypothesis_attack.T

    nt_a = int(n_attack * 0.8)

    """ Generate random hyperparameters """
    hp_values = generate_random_hyperparameters(model_type=model_type)
    hp_values["epochs"] = epochs

    callback_sca_metric = GetSCAMetric(dataset.x_attack[:nt_a], labels[:nt_a], hp_values["epochs"],
                                       dataset.correct_key_attack, loss,
                                       dataset.x_attack[nt_a:], labels[nt_a:])

    """ Create model from random hyperparameters """
    if model_type == "mlp":
        model = mlp(features, loss, hp_values)
    else:
        model = cnn(features, loss, hp_values)

    """ Train model """
    history = model.fit(dataset.x_attack[:nt_a], labels[:nt_a], epochs=hp_values["epochs"],
                        batch_size=hp_values["batch_size"], verbose=2,
                        callbacks=[callback_sca_metric])

    """ Get results from callback """
    key_rank_evolution_loss_train = callback_sca_metric.key_rank_evolution_loss_train,
    key_rank_evolution_corr_train = callback_sca_metric.key_rank_evolution_corr_train,
    key_rank_evolution_loss_val = callback_sca_metric.key_rank_evolution_loss_val,
    key_rank_evolution_corr_val = callback_sca_metric.key_rank_evolution_corr_val,
    objective_function_from_loss_train = callback_sca_metric.objective_function_from_loss_train,
    objective_function_from_loss_val = callback_sca_metric.objective_function_from_loss_val,
    objective_function_from_corr_train = callback_sca_metric.objective_function_from_corr_train,
    objective_function_from_corr_val = callback_sca_metric.objective_function_from_corr_val,
    objective_function_from_corr_train = callback_sca_metric.objective_function_from_corr_train,
    objective_function_from_corr_val = callback_sca_metric.objective_function_from_corr_val,
    loss_candidates_epochs_train = callback_sca_metric.loss_candidates_epochs_train,
    loss_candidates_epochs_val = callback_sca_metric.loss_candidates_epochs_val,
    corr_epochs_train = callback_sca_metric.corr_epochs_train,
    corr_epochs_val = callback_sca_metric.corr_epochs_val,


multi_output_regression()
