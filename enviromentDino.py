from gym import Env
from gym.spaces import Box
from gym.spaces.discrete import Discrete
from gym.utils import seeding
from utils import SCREEN_HEIGHT, SCREEN_WIDTH
import numpy as np
import random
import time
import game
import cv2 as cv
import pyautogui

def get_image():
    """
    Makes a screenshot of the dino's window and preprocess it.

    :return: preprocessed screenshot.
    :rtype: 3 dimensional NumPy array.
    """
    myScreenshot = pyautogui.screenshot(region=(0,372, 960, 248))
    myScreenshot.save('screenshot.png')
    image = cv.imread('screenshot.png')
    average = np.mean(image)
    rect, image = cv.threshold(image,120,255,cv.THRESH_BINARY_INV)
    if average<100.0:
        rect, image = cv.threshold(image,120,255,cv.THRESH_BINARY_INV)
    image = cv.erode(image,kernel=None,iterations=1)
    image = cv.dilate(image,kernel=None,iterations=1)
    # uncomment if you want to see the screenshots during the execution.
    image = cv.resize(image, (SCREEN_HEIGHT, SCREEN_WIDTH), interpolation=cv.INTER_AREA)
    
    
    return image

class Dino(Env):
    """
    Represents a Chrome Dino Game enviroment.
    """
    def __init__(self):
        """
        Creates a Chrome Dino Game enviroment.

        """
        self.reward = 0
        self.action_space = Discrete(2) # HÁ TRÊS AÇÕES
        self.observation_space = Box(low=0, high=255, shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8) # O ESPAÇO É UMA IMAGEM
        self.score = 0

    def step(self,action):
        """
        Represents a step in the enviroment.

        :param action: action taken in that step.
        :type action: int.
        :return: screenshot after the step, reward after the step, boolean to know if the game is over, debugging tool.
        :rtype: 3 dimensional NumPy array, int, boolean, dict.
        """
        done = False # DONE SIGNIFICA QUE O JOGO ACABOU, E NO CASO, ELE SÓ ACABA SE DER GAME OVER
        self.reward = 1
        # SÃO TRÊS AÇÕES POSSÍVEIS, PULAR, ABAIXAR, OU CONTINUAR CORRENDO SEM REALIZAR NENHUMA OUTRA AÇÃO
        if action == 1:
            game.Jumping()
            self.reward = 0
        # elif action == 1:
        #     game.Ducking()
        #     self.reward = 0
        # elif action == 2:
        #     game.Running()
        time.sleep(0.1)
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
        """
        Render the enviroment.

        :param mode: render mode.
        :type mode: string.
        :return: enviroment image.
        :rtype: 3 dimensional NumPy array.
        """
        image = get_image()
        return image
        
    def reset(self):
        """
        Resets the enviroment.

        :return: enviroment image.
        :rtype: 3 dimensional NumPy array, int, boolean, dict.
        """
        game.restart() #JOGO RESETA QUANDO DÁ GAMEOVER
        return get_image()
        #Outra forma de realizar bastava apertar a tecla espaço ou a tecla enter para reinicializar o jogo depois do gameover (Entretanto, essa é a forma mais rápida)
    
    def close(self):
        """
        Closes the enviroment.

        """
        game.dinoGame.Quit()