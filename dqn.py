import tensorflow as tf
from tensorflow.keras import Sequential
import numpy as np
from collections import deque
import random
import os
import csv


class Mario:
    def __init__(self,input_dim,output_dim,save_dir):
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.save_dir = save_dir
        self.save_every = 5e5
        self.gamma = 0.9
        self.batch_size = 32
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025)
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.memory = deque(maxlen=100000)
        self.counter =0
        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync



    def net_init(self,state,modelIndex):
        c, h, w = self.input_dim
        self.predict = Sequential([
          tf.keras.layers.Conv2D(32,8,4,activation='relu',input_shape=(c, h, w),data_format="channels_first"),
          tf.keras.layers.Conv2D(64, 4, 2, activation='relu',padding="VALID"),
          tf.keras.layers.Conv2D(64, 3, 1, activation='relu'),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(512, input_shape=(3136,), activation='relu'),
          tf.keras.layers.Dense(self.output_dim, input_shape=(512,), activation=None)])

        self.target = Sequential([
            tf.keras.layers.Conv2D(32, 8, 4, activation='relu', input_shape=(4, 84, 84), data_format="channels_first"),
            tf.keras.layers.Conv2D(64, 4, 2, activation='relu', padding="VALID"),
            tf.keras.layers.Conv2D(64, 3, 1, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, input_shape=(3136,), activation='relu'),
            tf.keras.layers.Dense(self.output_dim, input_shape=(512,), activation=None)])

        if modelIndex==1:
            # self.predict.summary()
            return self.predict(state)
        if modelIndex==2:
            self.target.summary()
            return self.target(state)

    def act(self,state):
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.output_dim)
        else:
            state = state.__array__()
            state = np.expand_dims(state, axis=0)
            action_values = self.net_init(state,1)
            action_idx = int(tf.math.argmax(action_values, axis=1))

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        self.memory.append((state, next_state, action, reward, done,))
    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(tf.stack, zip(*batch))
        return state, next_state, tf.squeeze(action), tf.squeeze(reward), tf.squeeze(done)
    def pred(self,states,action):
        prediction=self.net_init(states,1).numpy()[np.arange(0, self.batch_size), action]
        return prediction

    def pred_next(self,states):
        prediction = self.net_init(states,2)
        return prediction

    def save_log(self, step, quantity, filename):
        with open(os.path.join(self.save_dir/"log2", filename), 'a+') as fi:
            csv_w = csv.writer(fi, delimiter=',')
            csv_w.writerow([step, quantity])

    def sync_Q_target(self):
        t=self.predict.save_weights("./targetModels/myModel",save_format="tf")
        self.target.load_weights("./targetModels/myModel")

    def save(self):
        save_path= self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}t"
        self.predict.save_weights(save_path)
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    @tf.function
    def train_step(self, states, actions):
        with tf.GradientTape() as tape:
            tape.watch(states)
            loss = tf.keras.losses.huber(states, actions)

        gradients = tape.gradient(loss, self.predict.trainable_variables)
        # gradients = [tf.clip_by_norm(gradient, 10) for gradient in gradients]

        self.optimizer.apply_gradients(zip(gradients, self.predict.trainable_variables))
        return loss




    def train(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        if self.curr_step % self.save_every == 0:
            self.save()
        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None


        self.counter += 1
        state, next_state, action, reward, done = self.recall()
        current_action_qs =self.pred(state, action)
        next_action_qs = self.pred_next(next_state)
        best_action=tf.math.argmax(next_action_qs, axis=1)
        next_Q_1=self.net_init(next_state,2).numpy()[np.arange(0, self.batch_size), best_action]
        next_Q_2=(reward + (1 - done.numpy()) * self.gamma * next_Q_1)
        loss=self.train_step(next_Q_2,current_action_qs)
        # self.save_log(self.counter, np.mean(loss), "loss.csv")
        return (int(current_action_qs.mean()), loss)




