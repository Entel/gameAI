import pygame
import random
from pygame.locals import *

FPS = 25
fpsclock = pygame.time.Clock()

# output of CNN
MOVE_STAY = [1, 0 ,0]
MOVE_LEFT = [0, 1, 0]
MOVE_RIGHT = [0, 0, 1]

# define the player
class Player(pygame.sprite.Sprite):
	def __init__(self):
		  super(Player, self).__init__()
		  self.surf = pygame.Surface((50, 25))
		  self.surf.fill((255, 255, 255))
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
		self.surf = pygame.Surface((50, 5))
		self.surf.fill((255, 255, 255))
		self.rect = self.surf.get_rect(center = (random.randint(0, 400), 0))
		self.speed = random.randint(5, 30)

 	def update(self):
  		self.rect.move_ip(0, self.speed)
  		if self.rect.bottom < 0:
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
		self.ADDENEMY = pygame.USEREVENT + (random.randint(3, 5))
		#pygame.time.set_timer(self.ADDENEMY, 50)

		self.enemies = pygame.sprite.Group()
		self.all_sprites = pygame.sprite.Group()
		self.all_sprites.add(self.player)

	def run(self):
		running = True

		#main loop
		while running:
			for event in pygame.event.get():
				if event.type == KEYDOWN:
					if event.key == K_ESCAPE:
						running = False
				elif event.type == QUIT:
					running = False
				elif (event.type == self.ADDENEMY):
					new_enemy = Enemy()
					self.enemies.add(new_enemy)
					self.score = self.score + 1
					self.all_sprites.add(new_enemy)

			pressed_keys = pygame.key.get_pressed()
			self.player.update(pressed_keys)
			self.enemies.update()
			self.screen.fill((0,0,0))
			# draw the player to the screen
			for entity in self.all_sprites:
				self.screen.blit(entity.surf, entity.rect)
			# return the score
			if pygame.sprite.spritecollideany(self.player, self.enemies):
				print self.score
				self.score = 0
				
			screen_image = pygame.surfarray.array3d(pygame.display.get_surface())		 
			# update the display
			pygame.display.flip()
			fpsclock.tick(FPS)
		#return reward, screen_image
	
	# action of the ai
	def step(self, action):
		if self.add_enemy_step == 6:
			new_enemy = Enemy()
			self.enemies.add(new_enemy)
			self.score = self.score + 1
			self.all_sprites.add(new_enemy)
			self.add_enemy_step = 0
		self.player.update(action)
		self.add_enemy_step = self.add_enemy_step + 1
		
		self.enemies.update()
		self.screen.fill((0, 0, 0))
		# draw the player to the screen
		for entity in self.all_sprites:
			self.screen.blit(entity.surf, entity.rect)
		# return the score
		if pygame.sprite.spritecollideany(self.player, self.enemies):
			print "score:" + str(self.score)
			self.score = 0
			
		screen_image = pygame.surfarray.array3d(pygame.display.get_surface())		 
		# update the display
		pygame.display.flip()
		return self.score, screen_image

game = Game()
# game.run()
# while 1:
#	game.step([0,1,0])
