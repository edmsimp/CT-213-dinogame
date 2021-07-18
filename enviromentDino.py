from gym import Env
from gym.spaces import Box
from gym.spaces.discrete import Discrete
from gym.utils import seeding
import numpy as np
import random
import time
import game
import cv2 as cv
import pyautogui
SCREEN_HEIGHT = 84
SCREEN_WIDTH = 84




""" FUNCTION - GET IMAGE"""


def get_image():
    myScreenshot = pyautogui.screenshot(region=(0,272, 960, 248))
    myScreenshot.save('screenshot.png')
    image = cv.imread('screenshot.png')
    rect, image = cv.threshold(image,120,255,cv.THRESH_BINARY_INV)
    image = cv.erode(image,kernel=None,iterations=1)
    image = cv.dilate(image,kernel=None,iterations=1)
    image = cv.resize(image, (SCREEN_HEIGHT, SCREEN_WIDTH), interpolation=cv.INTER_AREA)
    cv.imshow('dino',image)

    cv.waitKey(1)
    return image

class Dino(Env):
    def __init__(self):
        self.reward = 0
        self.action_space = Discrete(2) # HÁ TRÊS AÇÕES
        self.observation_space = Box(low=0, high=255, shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8) # O ESPAÇO É UMA IMAGEM
        self.score = 0
    def step(self,action):

        done = False # DONE SIGNIFICA QUE O JOGO ACABOU, E NO CASO, ELE SÓ ACABA SE DER GAME OVER
        self.reward = 1
        # SÃO TRÊS AÇÕES POSSÍVEIS, PULAR, ABAIXAR, OU CONTINUAR CORRENDO SEM REALIZAR NENHUMA OUTRA AÇÃO
        if action == 1:
            game.Jumping()
            self.reward = 0
        elif action == 2:
            game.Ducking()
            self.reward = 0

        if game.gameOver() == True: # Se o jogo deu game over faz:

            self.reward = -10 # RECOMPENSA RUIM, POIS ELE PERDEU O JOGO
            done = True  # JOGO ACABOU POIS DEU GAMER OVER
            
        else: # Se não, continua
            self.reward = 1 # RECOMPENSA BOA, POIS ELE NÃO PERDEU O JOGO AINDA E ELE TÁ ANDANDO

        # PEGA A IMAGEM PIXELADA 
        observation = get_image()

        # APARENTEMENTE SERVE PARA DEBUGAR, MAS NÃO SEI O QUE ISSO SIGNIFICA
        info = {}
        
        return observation, self.reward, done, info

    def render(self, mode='rgb_array'):
        image = get_image()
        return image
        
    def reset(self):
        #JOGO RESETA QUANDO DÁ GAMEOVER
        game.restart()
        time.sleep(0.1)
        return get_image()
        #Outra forma de realizar bastava apertar a tecla espaço ou a tecla enter para reinicializar o jogo depois do gameover (Entretanto, essa é a forma mais rápida)
    def close(self):
        game.dinoGame.Quit()