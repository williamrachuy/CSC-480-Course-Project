# Flappy Birds AI
# Q Learning with Keras

# -----------------------------------------------------------------------------
#                                   Modules
# -----------------------------------------------------------------------------

from itertools import cycle
import random
import sys

import pygame
from pygame.locals import *

import math
from collections import deque
import time


# -----------------------------------------------------------------------------
#                                   Constants
# -----------------------------------------------------------------------------

# Learning Constants
LOAD            = True
LOADFILE        = "./save/qvalues.txt"
SAVEFILE        = "./save/qvalues.txt"
INFINITE        = True      # train infinitely
EPOCHS          = 1000      # number of games to train on
XDIV            = 10        # group XDIV number of pixels in the x direction
YDIV            = 10        # group YDIV number of pixels in the y direction
LEARNINGRATE    = 0.1       # learning rate, 0-1
DISCOUNT        = 0.9       # discount factor, 0-1

# Speed Constants
SLEEP           = 0
FPS             = 400    # originally 30

# Agent Constants
STATE_SIZE      = 3
ACTION_SIZE     = 2

# Flappy Bird Display
SCREENWIDTH     = 288
SCREENHEIGHT    = 512
BASEY           = SCREENHEIGHT * 0.79
PIPEGAPSIZE     = 100 # gap between upper and lower part of pipe

# Mapping Constants (pipe - bird)
DX = range(0, int(math.floor(SCREENWIDTH / 2 - SCREENWIDTH * 0.2) / XDIV + 1))
XLEN = len(DX)
YMIN = math.floor((PIPEGAPSIZE - BASEY) / YDIV)
DY = range(YMIN, math.floor((int(BASEY * 0.6 - PIPEGAPSIZE) + PIPEGAPSIZE) / YDIV) + 1)
YLEN = len(DY)
VMIN = -9
V = range(VMIN,9)
VLEN = len(V)

# image, sound and hitmask  dicts
IMAGES, SOUNDS, HITMASKS = {}, {}, {}

# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    ('assets/sprites/redbird-upflap.png',
    'assets/sprites/redbird-midflap.png',
    'assets/sprites/redbird-downflap.png'),
    # blue bird
    ('assets/sprites/bluebird-upflap.png',
    'assets/sprites/bluebird-midflap.png',
    'assets/sprites/bluebird-downflap.png'),
    # yellow bird
    ('assets/sprites/yellowbird-upflap.png',
    'assets/sprites/yellowbird-midflap.png',
    'assets/sprites/yellowbird-downflap.png')
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)


try:
    xrange
except NameError:
    xrange = range

    
# -----------------------------------------------------------------------------
#                           Agent Class
# -----------------------------------------------------------------------------

class DQNAgent:
    def __init__(self):
        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE
        self.memory = deque(maxlen=2000)
        self.alpha = LEARNINGRATE
        self.gamma = DISCOUNT
        self.qvalues = []
        if LOAD:
            self.load_qvalues()
        else:     
            self.zero_qvalues()

    def map(self, dx, dy, v):
        dx = (dx - (dx % XDIV)) / XDIV
        dy = (dy - (dy % YDIV)) / YDIV - YMIN 
        v = v - VMIN
        return int((dx * YLEN * VLEN) + (dy * VLEN) + v)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.appendleft((state, action, reward, next_state, done))

    def act(self, state):
        qstate = self.map(state[0], state[1], state[2])
        if self.qvalues[qstate][0] < self.qvalues[qstate][1]:
            return 1
        else:
            return 0

    def update_qvalues(self):
        for state, action, reward, next_state, done in self.memory:
            qstate = self.map(state[0], state[1], state[2])
            next_qstate = self.map(next_state[0], next_state[1], next_state[2])
            self.qvalues[qstate][action] = \
                    self.qvalues[qstate][action] + self.alpha * \
                    (reward + self.gamma * max(self.qvalues[next_qstate]) - \
                    self.qvalues[qstate][action])       
        self.memory.clear()
        return

    def zero_qvalues(self):
        i = 0;
        Qvalues = list()
        for x in DX:
            for y in DY:
                for z in V:
                    self.qvalues.append([0,0])
        return

    def load_qvalues(self):
        for text in open(LOADFILE).read().split():
            a,b = text.strip("()").split(',')
            self.qvalues.append([float(a), float(b)])
        return
    
    def save_qvalues(self):
        with open(SAVEFILE, 'w') as f:
            for val in self.qvalues:
                f.write("(%s,%s) " % (str(val[0]), str(val[1])))
        return


# -----------------------------------------------------------------------------
#                           Initialization
# -----------------------------------------------------------------------------

def init():
    global SCREEN, FPSCLOCK, agent
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption('Flappy Bird')

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # game over sprite
    IMAGES['gameover'] = \
        pygame.image.load('assets/sprites/gameover.png').convert_alpha()
        
    # message sprite for welcome screen
    IMAGES['message'] = \
        pygame.image.load('assets/sprites/message.png').convert_alpha()

    # base (ground) sprite
    IMAGES['base'] = \
        pygame.image.load('assets/sprites/base.png').convert_alpha()

    # sounds
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    SOUNDS['die']    = pygame.mixer.Sound('assets/audio/die' + soundExt)
    SOUNDS['hit']    = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    SOUNDS['point']  = pygame.mixer.Sound('assets/audio/point' + soundExt)
    SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    SOUNDS['wing']   = pygame.mixer.Sound('assets/audio/wing' + soundExt)

    # Learning Agent
    agent = DQNAgent()


# -----------------------------------------------------------------------------
#                           Random Game Settings 
# -----------------------------------------------------------------------------

def setGameSettings():
    # select random background sprites
    randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
    IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

    # select random player sprites
    randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
    IMAGES['player'] = (
        pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
        pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
        pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
    )

    # select random pipe sprites
    pipeindex = random.randint(0, len(PIPES_LIST) - 1)
    IMAGES['pipe'] = (
        pygame.transform.rotate(
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), 180),
        pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
    )

    # hismask for pipes
    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )

    # hitmask for player
    HITMASKS['player'] = (
        getHitmask(IMAGES['player'][0]),
        getHitmask(IMAGES['player'][1]),
        getHitmask(IMAGES['player'][2]),
    )


# -----------------------------------------------------------------------------
#                           Show Welcome Animation
# -----------------------------------------------------------------------------

def showWelcomeAnimation():
    # index of player to blit on screen
    playerIndex = 0
    playerIndexGen = cycle([0, 1, 2, 1])
    
    # iterator used to change playerIndex after every 5th iteration
    loopIter = 0

    playerx = int(SCREENWIDTH * 0.2)
    playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)

    messagex = int((SCREENWIDTH - IMAGES['message'].get_width()) / 2)
    messagey = int(SCREENHEIGHT * 0.12)

    basex = 0
    # amount by which base can maximum shift to left
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # player shm for up-down motion on welcome screen
    playerShmVals = {'val': 0, 'dir': 1}

    # Wait 1 seconds, then start
    t = time.time()
    while time.time() - t < SLEEP:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
        # adjust playery, playerIndex, basex
        if (loopIter + 1) % 5 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 4) % baseShift)
        playerShm(playerShmVals)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))
        SCREEN.blit(IMAGES['player'][playerIndex],
                    (playerx, playery + playerShmVals['val']))
        SCREEN.blit(IMAGES['message'], (messagex, messagey))
        SCREEN.blit(IMAGES['base'], (basex, BASEY))

        pygame.display.update()
        FPSCLOCK.tick(FPS)

    return {
        'playery': playery + playerShmVals['val'],
        'basex': basex,
        'playerIndexGen': playerIndexGen,
    }


# -----------------------------------------------------------------------------
#                               Main Game
# -----------------------------------------------------------------------------

def mainGame(movementInfo):
    score = playerIndex = loopIter = 0
    playerIndexGen = movementInfo['playerIndexGen']
    playerx, playery = int(SCREENWIDTH * 0.2), movementInfo['playery']
    basex = movementInfo['basex']
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH / 2, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH, 'y': newPipe2[0]['y']},
    ]

    # list of lowerpipe
    lowerPipes = [
        {'x': SCREENWIDTH / 2, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH, 'y': newPipe2[1]['y']},
    ]
 

    pipeVelX = -4

    # player velocity, max velocity, downward accleration, accleration on flap
    playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
    playerMaxVelY =  10   # max vel along Y, max descend speed
    playerMinVelY =  -8   # min vel along Y, max ascend speed
    playerAccY    =   1   # players downward accleration
    playerRot     =  45   # player's rotation
    playerVelRot  =   3   # angular speed
    playerRotThr  =  20   # rotation threshold
    playerFlapAcc =  -9   # players speed on flapping
    playerFlapped = False # True when player flaps

    # TODO TODO TODO
    # Initial game state
    deltaX = int((lowerPipes[0]['x'] - playerx) / 10);
    deltaY = int((lowerPipes[0]['y'] - playery) / 10);
    state = [deltaX, deltaY, playerVelY]
    # TODO TODO TODO

    while True:

        playerScored = False

        # Manual input
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                if playery > -2 * IMAGES['player'][0].get_height():
                    playerVelY = playerFlapAcc
                    playerFlapped = True
                    SOUNDS['wing'].play()

        # Get action from agent and jump if necessary
        action = agent.act(state)
        if action:
            if playery > -2 * IMAGES['player'][0].get_height():
                playerVelY = playerFlapAcc
                playerFlapped = True
                SOUNDS['wing'].play()

        # check for crash here
        crashTest = checkCrash({'x': playerx, 'y': playery, 'index': playerIndex},
                               upperPipes, lowerPipes)

        # Store crash info if the game is over
        if crashTest[0]:
            crashInfo = {
                'y': playery,
                'groundCrash': crashTest[1],
                'basex': basex,
                'upperPipes': upperPipes,
                'lowerPipes': lowerPipes,
                'score': score,
                'playerVelY': playerVelY,
                'playerRot': playerRot
            }

        # check for score
        playerMidPos = playerx + IMAGES['player'][0].get_width() / 2
        for pipe in upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                score += 1
                SOUNDS['point'].play()
                playerScored = True

        # TODO TODO TODO
        # Assign reward based on result
        if crashTest[0]:    # crashed
            reward = -10
            done = True
        else:
            reward = 1
            done = False
        # TODO TODO TODO

        # playerIndex basex change
        if (loopIter + 1) % 3 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        # rotate the player
        if playerRot > -90:
            playerRot -= playerVelRot

        # player's movement
        if playerVelY < playerMaxVelY and not playerFlapped:
            playerVelY += playerAccY
        if playerFlapped:
            playerFlapped = False

            # more rotation to cover the threshold (calculated in visible rotation)
            playerRot = 45

        playerHeight = IMAGES['player'][playerIndex].get_height()
        playery += min(playerVelY, BASEY - playery - playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        # print score so player overlaps the score
        showScore(score)

        # Player rotation has a threshold
        visibleRot = playerRotThr
        if playerRot <= playerRotThr:
            visibleRot = playerRot
        
        # TODO TODO TODO 
        # Get the next game state
        deltaX = int((lowerPipes[0]['x'] - playerx) / 10);
        deltaY = int((lowerPipes[0]['y'] - playery) / 10);
        if deltaX < 0:
            deltaX = int((lowerPipes[1]['x'] - playerx) / 10);
            deltaY = int((lowerPipes[1]['y'] - playery) / 10);
        next_state = [deltaX, deltaY, playerVelY]
        # TODO TODO TODO

        # TODO TODO TODO
        # record the state
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        # TODO TODO TODO

        # Return if crashed
        if crashTest[0]:
            return crashInfo

        playerSurface = pygame.transform.rotate(IMAGES['player'][playerIndex], visibleRot)
        SCREEN.blit(playerSurface, (playerx, playery))

        pygame.display.update()
        FPSCLOCK.tick(FPS)


# -----------------------------------------------------------------------------
#                           Show Game Over Screen
# -----------------------------------------------------------------------------

def showGameOverScreen(crashInfo):
    """crashes the player down ans shows gameover image"""
    score = crashInfo['score']
    playerx = SCREENWIDTH * 0.2
    playery = crashInfo['y']
    playerHeight = IMAGES['player'][0].get_height()
    playerVelY = crashInfo['playerVelY']
    playerAccY = 2
    playerRot = crashInfo['playerRot']
    playerVelRot = 7

    basex = crashInfo['basex']

    upperPipes, lowerPipes = crashInfo['upperPipes'], crashInfo['lowerPipes']

    # play hit and die sounds
    SOUNDS['hit'].play()
    if not crashInfo['groundCrash']:
        SOUNDS['die'].play()

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
                
        if playery + playerHeight >= BASEY - 1:
            time.sleep(SLEEP)
            return score

        # player y shift
        if playery + playerHeight < BASEY - 1:
            playery += min(playerVelY, BASEY - playery - playerHeight)

        # player velocity change
        if playerVelY < 15:
            playerVelY += playerAccY

        # rotate only when it's a pipe crash
        if not crashInfo['groundCrash']:
            if playerRot > -90:
                playerRot -= playerVelRot

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        showScore(score)

        playerSurface = pygame.transform.rotate(IMAGES['player'][1], playerRot)
        SCREEN.blit(playerSurface, (playerx,playery))

        FPSCLOCK.tick(FPS)
        pygame.display.update()

    time.sleep(SLEEP)
    return score


# -----------------------------------------------------------------------------
#                               Helper Functions
# -----------------------------------------------------------------------------

def playerShm(playerShm):
    """oscillates the value of playerShm['val'] between 8 and -8"""
    if abs(playerShm['val']) == 8:
        playerShm['dir'] *= -1

    if playerShm['dir'] == 1:
         playerShm['val'] += 1
    else:
        playerShm['val'] -= 1


def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE}, # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return [True, True]
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])
        pipeW = IMAGES['pipe'][0].get_width()
        pipeH = IMAGES['pipe'][0].get_height()

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return [True, False]

    return [False, False]

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in xrange(rect.width):
        for y in xrange(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in xrange(image.get_width()):
        mask.append([])
        for y in xrange(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask


def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

# -----------------------------------------------------------------------------
#                               Training
# -----------------------------------------------------------------------------

def train():
    e = 0
    while e < EPOCHS or INFINITE:
        # play game
        setGameSettings()
        movementInfo = showWelcomeAnimation()
        crashInfo = mainGame(movementInfo)
        score = showGameOverScreen(crashInfo)

        # Train agent on last game
        agent.update_qvalues()

        if INFINITE:
            print("epoch: {}/inf, score: {}"
                .format(e, score))
        else:
            print("epoch: {}/{}, score: {}"
                .format(e, EPOCHS, score))
        
        if e % 10 == 0:
            agent.save_qvalues()

        e+=1


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    init() 
    train()

