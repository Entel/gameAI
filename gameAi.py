import pygame
import random
from pygame.locals import *
import numpy as np
from collections import deque
import tensorflow as tf
import cv2

FPS = 25
fpsclock = pygame.time.Clock()

# output of CNN
MOVE_STAY = [1, 0 ,0]
MOVE_LEFT = [0, 1, 0]
MOVE_RIGHT = [0, 0, 1]
SCORE = []

# define the player
class Player(pygame.sprite.Sprite):
	def __init__(self):
		  super(Player, self).__init__()
		  self.surf = pygame.Surface((30, 30))
		  self.surf.fill((123, 55, 25))
		  self.rect = self.surf.get_rect()
		  self.rect.y = 550
		  self.rect.x = 175
 
 	def update(self, action):
		"""
		if pressed_keys[K_UP]:
			self.rect.move_ip(0, -5)
		if pressed_keys[K_DOWN]:
			self.rect.move_ip(0, 5)
		"""
		if action == MOVE_LEFT:
			self.rect.move_ip(-5, 0)
		if action == MOVE_RIGHT:
			self.rect.move_ip(5, 0)
		if action == MOVE_STAY:
			self.rect.move_ip(0, 0)

  		#keep player on the screen
  		if self.rect.left < 0:
   			self.rect.left = 0
	  	elif self.rect.right > 400:
   			self.rect.right = 400

class Enemy(pygame.sprite.Sprite):
 	def __init__(self):
		super(Enemy, self).__init__()
		self.surf = pygame.Surface((20, 20))
		self.surf.fill((255, 255, 255))
		self.rect = self.surf.get_rect(center = (random.randint(10, 390), 0))
		self.speed = random.randint(5, 8)

 	def update(self):
  		self.rect.move_ip(0, self.speed)
  		if self.rect.bottom <= 0:
   			self.kill()

class Game():
	def __init__(self):
		#initialization
		pygame.init()

		#create the screen object
		self.screen = pygame.display.set_mode((400, 600))

		#instantiate the player
		self.player = Player()
		self.score = 0

		#create enemies
		self.add_enemy_step = 0
		#self.ADDENEMY = pygame.USEREVENT + (random.randint(2, 4))
		self.ADDENEMY = pygame.USEREVENT + 1
		pygame.time.set_timer(self.ADDENEMY, 50)

		self.enemies = pygame.sprite.Group()
		self.all_sprites = pygame.sprite.Group()
		self.all_sprites.add(self.player)

	# action of the AI
	def step(self, action):
		if self.add_enemy_step == 15:
			self.add_enemy_step = 0
			new_enemy = Enemy()
			self.enemies.add(new_enemy)
			self.score = self.score + 1
			self.all_sprites.add(new_enemy)
		self.player.update(action)
		self.add_enemy_step = self.add_enemy_step + 1	

		self.enemies.update()
		self.screen.fill((0, 0, 0))
		# draw the player to the screen
		for entity in self.all_sprites:
			self.screen.blit(entity.surf, entity.rect)
		# return the score
		if pygame.sprite.spritecollideany(self.player, self.enemies):
			#print "score:" + str(self.score)
			SCORE.append(self.score)
			self.score = 0

		pygame.display.flip()
		screen_image = pygame.surfarray.array3d(pygame.display.get_surface())		 
		# update the display
		return self.score, screen_image

# game = Game()
# game.run()
# while 1:
#	game.step([0,1,0])

LEARNING_RATE = 0.99
INITIAL_EPSLON = 1.0
FINAL_EPSILON = 0.05
EXPLORE = 500000
OBSERVE = 50000
REPLAY_MEMORY = 500000
BATCH = 100

output = 3
input_image = tf.placeholder("float", [None, 100, 150, 4])
action = tf.placeholder("float", [None, output])

# define the CNN
def convolutional_neural_network(input_image):
	weights = {'w_conv1': tf.Variable(tf.zeros([8, 8, 4, 32])),
		'w_conv2': tf.Variable(tf.zeros([4, 4, 32, 64])),
		'w_conv3': tf.Variable(tf.zeros([3, 3, 64, 64])),
		'w_fc4': tf.Variable(tf.zeros([8640, 784])),
		'w_out': tf.Variable(tf.zeros([784, output]))}
	biases = {'b_conv1': tf.Variable(tf.zeros([32])),
		'b_conv2': tf.Variable(tf.zeros([64])),
		'b_conv3': tf.Variable(tf.zeros([64])),
		'b_fc4': tf.Variable(tf.zeros([784])),
		'b_out': tf.Variable(tf.zeros([output]))}

	conv1 = tf.nn.relu(tf.nn.conv2d(input_image, weights['w_conv1'], strides = [1, 4, 4, 1], padding = "VALID") + biases['b_conv1'])
	conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['w_conv2'], strides = [1, 2, 2, 1], padding = "VALID") + biases['b_conv2'])
	conv3 = tf.nn.relu(tf.nn.conv2d(conv2, weights['w_conv3'], strides = [1, 1, 1, 1], padding = "VALID") + biases['b_conv3'])
	conv3_flat = tf.reshape(conv3, [-1, 8640])
	fc4 = tf.nn.relu(tf.matmul(conv3_flat, weights['w_fc4']) + biases['b_fc4'])
	output_layer = tf.matmul(fc4, weights['w_out']) + biases['b_out']
	return output_layer

def train_neural_network(input_image):
	predict_action = convolutional_neural_network(input_image)
	
	argmax = tf.placeholder("float", [None, output])
	gt = tf.placeholder("float", [None])

	action = tf.reduce_sum(tf.mul(predict_action, argmax), reduction_indices = 1)
	cost = tf.reduce_mean(tf.square(action - gt))
	optimizer = tf.train.AdamOptimizer(1e-6).minimize(cost)

	game = Game()
	D = deque()

	_, image = game.step(MOVE_STAY)

	image = cv2.cvtColor(cv2.resize(image, (150, 100)), cv2.COLOR_BGR2GRAY)
	ret, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
	input_image_data = np.stack((image, image, image, image), axis = 2)

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
	
		saver = tf.train.Saver()

		n = 0
		epsilon = INITIAL_EPSLON
		while True:
			action_t = predict_action.eval(feed_dict = {input_image : [input_image_data]})[0]
			
			argmax_t =np.zeros([output], dtype = np.int)
			if (random.random() <= INITIAL_EPSLON):
				maxIndex = random.randrange(output)
			else:
				maxIndex = np.argmax(action_t)
			argmax_t[maxIndex] = 1
			if epsilon > FINAL_EPSILON:
				epsilon -= (INITIAL_EPSLON - FINAL_EPSILON) / EXPLORE

			reward, image = game.step(list(argmax_t))
			
			image = cv2.cvtColor(cv2.resize(image, (100, 150)), cv2.COLOR_BGR2GRAY)
			ret, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
			image = np.reshape(image, (100, 150, 1))
			input_image_data1 = np.append(image, input_image_data[:, :, 0:3], axis = 2)

			D.append((input_image_data, argmax_t, reward, input_image_data1))

			if len(D) > REPLAY_MEMORY:
				D.popleft()
			
			if n > OBSERVE:
				minibatch = random.sample(D, BATCH)
				input_image_data_batch = [d[0] for d in minibatch]
				argmax_batch = [d[1] for d in minibatch]
				reward_batch = [d[2] for d in minibatch]
				input_image_data1_batch = [d[3] for d in minibatch]

				gt_batch = []

				out_batch = predict_action.eval(feed_dict = {input_image : input_image_data1_batch})
				
				for i in range(0, len(minibatch)):
					gt_batch.append(reward_batch[i] + LEARNING_RATE * np.max(out_batch[i]))

				optimizer.run(feed_dict = {gt : gt_batch, argmax : argmax_batch, input_image : input_image_data_batch})

			input_image_data = input_image_data1
			n = n + 1
	
			#if n % 10000 == 0:
			#	saver.save(sess, 'game.cpk', global_step = n)

			print(n, "epsilon:", epsilon, " " ,"action:", maxIndex, " " ,"reward:", reward)
 

train_neural_network(input_image)

