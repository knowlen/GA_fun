# Author: Nick Knowles (knowlen@wwu.edu)
# Date: Feb 22, 2017
# This is a niave baseline implimentation for a generative machine learning model that 
# attempts to approximate a given image using an evolutionary algorithm.

import scipy.misc as sci
import numpy as np
import random

label_shape = [450,450,3]


class candidate:
    global label_shape
    def __init__(self):
        self.fitness = 0
        self.img = (np.random.randint(255,size=label_shape))


# size = the population sample size.
# champions = high fitness candidates from last evaluation.
# x = the number of pixel rows in the image
# y = the number of pixel cols in the image
#
class population:
    def __init__(self, size, champions):
    # look at hr sk sample code. 
        a_size = size-len(champions) #adjusted size
        self.pop = [candidate() for i in xrange(a_size)]
# [(np.full(shape=(240,240, 4), fill_value=(int(255*random.random())),
#                     dtype=int)) for i in xrange(a_size)]
#[[[int(255*random.random()) for k in xrange(a_size)] for j in xrange(a_size)] for i in xrange(4)]
       # [int(255*random.random()) for i in xrange(size-len(champions))]
        self.champions = champions
        for champ in champions:
            self.pop.append(champ)
            
    # needed? 
    #def __iter__(self):
    #    return (i for i in (self.pop, self.champions))

    def eval(self, label):
        # for c in self.pop
        #   
        for can in self.pop:
            error = np.sum(np.absolute(np.subtract(label, can.img)))
            can.fitness = 1.0/error
             
    
    def crossover():
        
        return
   
# --START--
#   Initialize champion list, c.
#   Load the target image, label.
#   Initialize a new population object, p.
#

label = sci.imread('/home/knowlen/Pictures/hutch_research.png')
c = [candidate()]

p = population(50, c) 
'''
for i in xrange(450):
    for j in xrange(450):
        for k in xrange(3):
            c[0].img[i][j][k] = #(random.random())*label[i][j][k]

'''

p.eval(label)
for i in p.pop:
    print i.fitness


#sci.imsave('hr_adversary1.png', c[0].img)
count = 0 




