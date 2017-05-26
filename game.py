import pygame
import random
from pygame.locals import *

FPS = 25
fpsclock = pygame.time.Clock()

#define the player
class Player(pygame.sprite.Sprite):
	def __init__(self):
		  super(Player, self).__init__()
		  self.surf = pygame.Surface((30, 30))
		  self.surf.fill((123, 123, 123))
		  self.rect = self.surf.get_rect()
		  self.rect.y = 550
		  self.rect.x = 175
 
 	def update(self, pressed_keys):
		"""
		if pressed_keys[K_UP]:
			self.rect.move_ip(0, -5)
		if pressed_keys[K_DOWN]:
			self.rect.move_ip(0, 5)
		"""
		if pressed_keys[K_LEFT]:
			self.rect.move_ip(-5, 0)
		if pressed_keys[K_RIGHT]:
			self.rect.move_ip(5, 0)

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
		self.rect = self.surf.get_rect(center = (random.randint(0, 400), 0))
		self.speed = random.randint(5, 10)

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
		self.ADDENEMY = pygame.USEREVENT + 2
		pygame.time.set_timer(self.ADDENEMY, 250)

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
			#draw the player to the screen
			for entity in self.all_sprites:
				self.screen.blit(entity.surf, entity.rect)
			if pygame.sprite.spritecollideany(self.player, self.enemies):
				print 'score: ' + str(self.score)
				self.score = 0
				
			screen_image = pygame.surfarray.array3d(pygame.display.get_surface())		 
			#update the display
			pygame.display.update()
			#fpsclock.tick(FPS)
		#return reward, screen_image

game = Game()
game.run()
