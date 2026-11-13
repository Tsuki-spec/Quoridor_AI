"""
author: cms
date: 2026/10/11
File introduction: TensorFlow / Keras residual network with policy and value heads.
"""

import numpy as np
import tensorflow as tf
from game import ACTION_SIZE
from search import board_paths


class PolicyValueNet:
    def __init__(self, learning_rate = 0.001, residual_blocks = 4, channels = 64, distance_features = True, race_features = True):
        if any(not isinstance(number, int) or number <= 0 for number in (residual_blocks, channels)):
            raise ValueError("residual_blocks 和 channels 必须是正整数")

        # Reduce numerical differences between single-state and batched inference.
        tf.config.experimental.enable_tensor_float_32_execution(False)
        self.distance_features = bool(distance_features)
        self.race_features = self.distance_features and bool(race_features)
        self.residual_blocks = residual_blocks
        self.channels = channels
        inputs = tf.keras.Input(shape = (9, 9, 9 if self.distance_features else 7))
        x = tf.keras.layers.Conv2D(
            channels, 3, padding = "same", use_bias = False,
            kernel_initializer = "he_normal", name = "stem_conv"
        )(inputs)
        x = tf.keras.layers.BatchNormalization(name = "stem_bn")(x)
        x = tf.keras.layers.ReLU(name = "stem_relu")(x)

        for index in range(residual_blocks):
            shortcut = x
            x = tf.keras.layers.Conv2D(
                channels, 3, padding = "same", use_bias = False,
                kernel_initializer = "he_normal", name = f"block_{index}_conv1"
            )(x)
            x = tf.keras.layers.BatchNormalization(name = f"block_{index}_bn1")(x)
            x = tf.keras.layers.ReLU(name = f"block_{index}_relu1")(x)
            x = tf.keras.layers.Conv2D(
                channels, 3, padding = "same", use_bias = False,
                kernel_initializer = "he_normal", name = f"block_{index}_conv2"
            )(x)
            x = tf.keras.layers.BatchNormalization(name = f"block_{index}_bn2")(x)
            x = tf.keras.layers.Add(name = f"block_{index}_add")([shortcut, x])
            x = tf.keras.layers.ReLU(name = f"block_{index}_relu2")(x)

        policy = tf.keras.layers.Conv2D(2, 1, use_bias = False, kernel_initializer = "he_normal", name = "policy_conv")(x)
        policy = tf.keras.layers.BatchNormalization(name = "policy_bn")(policy)
        policy = tf.keras.layers.ReLU(name = "policy_relu")(policy)
        policy = tf.keras.layers.Flatten(name = "policy_flatten")(policy)
        policy = tf.keras.layers.Dense(ACTION_SIZE, name = "policy")(policy)

        value = tf.keras.layers.Conv2D(1, 1, use_bias = False, kernel_initializer = "he_normal", name = "value_conv")(x)
        value = tf.keras.layers.BatchNormalization(name = "value_bn")(value)
        value = tf.keras.layers.ReLU(name = "value_relu")(value)
        value = tf.keras.layers.Flatten(name = "value_flatten")(value)
        if self.race_features:
            race = tf.keras.layers.Lambda(lambda state: tf.stack([
                tf.reduce_sum(state[:, :, :, 0] * state[:, :, :, 7], axis = (1, 2)),
                tf.reduce_sum(state[:, :, :, 1] * state[:, :, :, 8], axis = (1, 2)),
                state[:, 0, 0, 4], state[:, 0, 0, 5]
            ], axis = 1), name = "race_features")(inputs)
            value = tf.keras.layers.Concatenate(name = "value_features")([value, race])
        value = tf.keras.layers.Dense(128, activation = "relu", name = "value_hidden")(value)
        value = tf.keras.layers.Dense(1, activation = "tanh", name = "value")(value)

        self.model = tf.keras.Model(inputs, [policy, value])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        # Keras 2 compatibility.
        if hasattr(self.optimizer, "jit_compile"):
            self.optimizer.jit_compile = False
        self.optimizer.build(self.model.trainable_variables)

    def predict(self, state):
        logits, values = self.predict_batch([state])
        return logits[0], float(values[0])

    @tf.function(input_signature = [tf.TensorSpec(shape = (None, 9, 9, None), dtype = tf.float32)], jit_compile = False)
    def inference(self, batch):
        # Use saved BatchNorm statistics.
        return self.model(batch, training = False)

    def encode_states(self, states):
        batch = np.asarray(states, dtype = np.float32)
        if batch.ndim != 4 or batch.shape[1:3] != (9, 9):
            raise ValueError("状态必须是 (batch, 9, 9, channels) 数组")
        channels = 9 if self.distance_features else 7
        if batch.shape[-1] == channels:
            return batch
        if not self.distance_features or batch.shape[-1] != 7:
            raise ValueError("状态通道数与网络不匹配")
        result = np.zeros((*batch.shape[:3], 9), dtype = np.float32)
        result[:, :, :, :7] = batch
        for index, state in enumerate(batch):
            walls = [frozenset((int(row), int(col)) for row, col in np.argwhere(state[:8, :8, plane] > 0.5))
                     for plane in (2, 3)]
            _, distances = board_paths(*walls)
            own = 0 if state[0, 0, 6] > 0.5 else 1
            result[index, :, :, 7] = np.asarray(distances[own], dtype = np.float32).reshape(9, 9) / 9.0
            result[index, :, :, 8] = np.asarray(distances[1 - own], dtype = np.float32).reshape(9, 9) / 9.0
        return result

    def predict_batch(self, states):
        batch = tf.convert_to_tensor(self.encode_states(states))
        logits, values = self.inference(batch)
        return logits.numpy(), values.numpy()[:, 0]

    def train_batch(self, states, policies, values, return_losses = False):
        states = tf.convert_to_tensor(self.encode_states(states))
        policies = tf.convert_to_tensor(np.asarray(policies, dtype = np.float32))
        values = tf.convert_to_tensor(np.asarray(values, dtype = np.float32).reshape(-1, 1))
        losses = self.training_step(states, policies, values)
        losses = tuple(float(loss.numpy()) for loss in losses)
        return losses if return_losses else losses[0]

    @tf.function(input_signature = [
        tf.TensorSpec(shape = (None, 9, 9, None), dtype = tf.float32),
        tf.TensorSpec(shape = (None, ACTION_SIZE), dtype = tf.float32),
        tf.TensorSpec(shape = (None, 1), dtype = tf.float32)
    ], jit_compile = False)
    def training_step(self, states, policies, values):
        with tf.GradientTape() as tape:
            logits, predictions = self.model(states, training = True)
            policy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = policies, logits = logits))
            value_loss = tf.reduce_mean(tf.square(values - predictions))
            loss = policy_loss + value_loss
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # Reject non-finite loss before updating weights.
        with tf.control_dependencies([tf.debugging.check_numerics(loss, "训练损失出现 NaN 或无穷大")]):
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, policy_loss, value_loss

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        try:
            self.model.load_weights(path)
            return
        except ValueError as error:
            original_error = error
        if self.distance_features:
            for use_maps in (False, True):
                legacy = PolicyValueNet(residual_blocks = self.residual_blocks, channels = self.channels,
                                        distance_features = use_maps, race_features = False)
                try:
                    legacy.model.load_weights(path)
                except ValueError:
                    continue
                # Zero added kernels preserve legacy predictions before fine-tuning.
                for layer in legacy.model.layers:
                    weights = layer.get_weights()
                    if not weights:
                        continue
                    target = self.model.get_layer(layer.name)
                    if layer.name in ("stem_conv", "value_hidden") and weights[0].shape != target.get_weights()[0].shape:
                        kernel = np.zeros(target.get_weights()[0].shape, dtype = np.float32)
                        if layer.name == "stem_conv":
                            kernel[:, :, :weights[0].shape[2], :] = weights[0]
                        else:
                            kernel[:weights[0].shape[0], :] = weights[0]
                        weights[0] = kernel
                    target.set_weights(weights)
                return
        raise ValueError(f"权重与当前残差网络不匹配：{path}；请使用相同结构的权重。") from original_error
