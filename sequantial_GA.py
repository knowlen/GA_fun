#Author: Nick Knowles (knowlen@wwu.edu)
# Date: Feb 22, 2017
# This is a niave baseline implimentation for a generative machine learning model that 
# attempts to approximate a given image using an evolutionary algorithm.

import scipy.misc as sci
import numpy as np
import random

label_shape = [450,450,3]


class candidate:
    global label_shape
    def __init__(self, init_state=""):
        self.fitness = 0
        if init_state == "blank":
            self.img = None
        else:
            self.img = (np.random.randint(255,size=label_shape))


# size = the population sample size.
# champions = high fitness candidates from last evaluation.
# x = the number of pixel rows in the image
# y = the number of pixel cols in the image
#
class population:
    def __init__(self, size):
    # look at hr sk sample code. 
        #a_size = size-len(champions) #adjusted size
        self.pop = [candidate() for i in xrange(size)]
# [(np.full(shape=(240,240, 4), fill_value=(int(255*random.random())),
#                     dtype=int)) for i in xrange(a_size)]
#[[[int(255*random.random()) for k in xrange(a_size)] for j in xrange(a_size)] for i in xrange(4)]
       # [int(255*random.random()) for i in xrange(size-len(champions))]
        self.children = []
        #for champ in champions:
        #    self.pop.append(champ)
            
    # needed? 
    #def __iter__(self):
    #    return (i for i in (self.pop, self.champions))



def evaluate(sample, label):
    # for c in self.pop
    #  concurrency would go here
    for can in sample:
        error = np.sum(np.absolute(np.subtract(label, can.img)))
        can.fitness = 1.0/error
         
    
def crossover(p_a, p_b):
    mask_a = np.random.choice(2, size=(450,450,3))
    mask_b = (mask_a - 1) * - 1
    c_a = candidate("blank") 
    c_a.img = (p_a.img * mask_a) + (p_b.img * mask_b)
    c_b = candidate("blank")
    c_b.img = (p_a.img * mask_b) + (p_b.img * mask_a)
    return c_a, c_b
   

def mutate(c):
    #look into mutation techniques     
    pass
    

def tournament_select(pop, t_size, k):
    # Tournament Selection
    # Apply probability dist if time:
    #   -sort on fitness
    selected = candidate()
    for i in xrange(k):
        sample = random.sample(pop, t_size)
        for c in sample:
            if selected.fitness < c.fitness:
                selected = c
            elif selected.fitness == c.fitness:
                selected = random.choice([c, selected]) #might not need the []
     
    return selected 


def replacement():
    # truncated: take best N from children & pop
    # elitest: take a few best from pop, rest children
    # generational: new p.pop = p.children

    pass

## ARGPARSE HERE
# --START--
#   Initialize champion list, c.
#   Load the target image, label.
#   Initialize a new population object, p.
#


label = sci.imread('/home/knowlen/Pictures/hutch_research.png')
c = candidate()

p = population(50) 
evaluate(p.pop, label)
p.children.append(c)
# random sample t_size candidates from population
# argparse this parameter later
t_size = 5

parent_a = tournament_select(p.pop, 2, 2)
parent_b = tournament_select(p.pop, 2, 2)
child_a, child_b = crossover(parent_a, parent_b)
# EVAL CHILD FITNESS HERE
# when you change impliment truncated replacement
#p.pop = p.children
#for i in sample:
#    print i.fitness

#sci.imsave('hr_adversary1.png', c[0].img)
count = 0 




