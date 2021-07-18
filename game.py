from pyscreeze import GRAYSCALE_DEFAULT
from selenium.webdriver.common import keys
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
import random
import time
import numpy as np




class DinoGame:
    def __init__(self):
        # ATENÇÃO!!! MUDE O DIRETÓRIO CASO NÃO RODE!
        self.driver_path = "chromedriver.exe"
        
        self.options = webdriver.ChromeOptions()
        self.options.add_argument(
            "--user-data-dir=/home/username/.config/google-chrome")
        self.chrome = webdriver.Chrome(
            self.driver_path,
            options=self.options
        
        )
    
    def LinkDino(self):
        # Acessa um site do dino
        try:
            self.chrome.get('chrome://dino/')
        except:
            print("Error")
    def Quit(self):
        self.chrome.quit()



"ACTIONS"

NUM_ACTIONS = 2

def Jumping():
    webdriver.ActionChains(dinoGame.chrome).key_down(Keys.SPACE).perform()
    webdriver.ActionChains(dinoGame.chrome).key_up(Keys.SPACE).perform()
     

def Ducking():
    webdriver.ActionChains(dinoGame.chrome).key_down(Keys.DOWN).perform()
   
    webdriver.ActionChains(dinoGame.chrome).key_up(Keys.DOWN).perform()   

def gameOver():
    return dinoGame.chrome.execute_script("return Runner.instance_.crashed")
def velocity():
    return dinoGame.chrome.execute_script("return Runner.instance_.currentSpeed")
def restart():
    return dinoGame.chrome.execute_script("Runner.instance_.restart();")

  
dinoGame = DinoGame()
dinoGame.LinkDino()
    
     
        
    