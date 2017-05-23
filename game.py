import pygame
from pygame.locals import *

#define the player
class Player(pygame.sprite.Sprite):
 def __init__(self):
  super(Player, self).__init__()
  self.surf = pygame.Surface((50, 25))
  self.surf.fill((255,255,255))
  self.rect = self.surf.get_rect()
 
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

#initialization
pygame.init()

#create the screen object
screen = pygame.display.set_mode((400, 600))

#instantiate the player
player = Player()

running = True

#main loop
while running:
 for event in pygame.event.get():
  if event.type == KEYDOWN:
   if event.key == K_ESCAPE:
    running = False
  elif event.type == QUIT:
   running = False

 pressed_keys = pygame.key.get_pressed()
 
 player.update(pressed_keys)

 #draw the player to the screen
 screen.blit(player.surf, (200, 550))
 #update the display
 pygame.display.flip()
