from selenium.webdriver.common.keys import Keys
from selenium import webdriver
import numpy as np
from tensorflow.python.ops.functional_ops import While


class DinoGame:
    """
    Represents a Chrome Dino Game itself.
    """
    def __init__(self):
        """
        Creates a Chrome Dino Game.

        """
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
        """
        Acess the Chrome Dino Game website.

        """
        try:
            self.chrome.get('chrome://dino/')
        except:
            print("Error")

    def Quit(self):
        """
        Quit the Chrome Dino Game website.

        """
        self.chrome.quit()

def Jumping():
    """
    Represents the jump action in the Chrome Dino Game.

    """
    webdriver.ActionChains(dinoGame.chrome).key_down(Keys.SPACE).perform()
    webdriver.ActionChains(dinoGame.chrome).key_up(Keys.SPACE).perform()     

def Ducking():
    """
    Represents the duck action in the Chrome Dino Game.

    """
    webdriver.ActionChains(dinoGame.chrome).key_down(Keys.DOWN).perform()   
    webdriver.ActionChains(dinoGame.chrome).key_up(Keys.DOWN).perform()   

def Running():
    """
    Dino keep Running
    """
    pass
def gameOver():
    """
    Returns wheter the Chrome Dino Game is over or not.

    :return: wheter the Chrome Dino Game is over or not.
    :rtype: boolean.
    """
    return dinoGame.chrome.execute_script("return Runner.instance_.crashed")

def velocity():
    """
    Returns the Chrome Dino Game velocity.

    :return: Chrome Dino Game velocity.
    :rtype: float.
    """
    return dinoGame.chrome.execute_script("return Runner.instance_.currentSpeed")

def restart():
    """
    Restarts the Chrome Dino Game.

    :return: Restart action.
    :rtype: script.
    """
    return dinoGame.chrome.execute_script("Runner.instance_.restart();")

def noAccelerate():
    dinoGame.chrome.execute_script("Runner.instance_.config.ACCELERATION = 0") 

def get_score():
    score = dinoGame.chrome.execute_script('return Runner.instance_.distanceMeter.digits;');
    return int((''.join(score)))
    


dinoGame = DinoGame()
dinoGame.LinkDino()
Jumping()
#noAccelerate()


    