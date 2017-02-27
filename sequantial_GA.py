#Author: Nick Knowles (knowlen@wwu.edu)
# Date: Feb 22, 2017
# This is a niave baseline implimentation for a generative machine learning model that 
# attempts to approximate a given image using an evolutionary algorithm.
# NOTES: pip installed futures into tf12GPU venv
import scipy.misc as sci
import numpy as np
import random, time, threading
import concurrent.futures
label_shape = [450,450,3]

class Semaphore(threading._Semaphore):
    wait = threading._Semaphore.acquire
    signal = threading._Semaphore.release

class Thread(threading.Thread):
    def __init__(self, t, *args):
        threading.Thread.__init__(self, target=t, args=args)
        self.start()

class candidate:
    global label_shape
    def __init__(self, init_state=""):
        self.fitness = 0
        if init_state == "blank":
            self.img = np.empty(label_shape)
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
   

def mutate(child):
    #look into mutation techniques     
    mask = np.random.choice(2, size=(450,450,3))
    neg_mask = -np.random.choice(2, size=(450,450,3))
    scalars = np.random.rand(450,450,3) #* mask) * neg_mask
	#seq element-wise multiplication might be better here
    child.img = child.img + (((child.img*scalars) * mask) * neg_mask)
    

def tournament_select(pop, t_size, k):
    # Tournament Selection
    # Look into applying probability dist if time:
    #   -sort on fitness
    #   -make a new sample_list, give most fit P entries, 2nd most 1-p^2 prob entries, ect..
    selected = candidate()
    for i in xrange(1):
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
# random sample t_size candidates from population
# argparse this parameter later
t_size = 10 
p = population(50) 
#p.children.append(c)
epoch = 0
evaluate(p.pop, label)
for i in p.pop:
    print i.fitness
while epoch < 100: 
    #for c in p.pop:
#	Thread(evaluate, [c], label)
    evaluate(p.pop, label)
    #with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    #	future_to_url = {executor.submit(evaluate, c, label): c for c in p.pop}

    for i in xrange(25):
        parent_a = tournament_select(p.pop, t_size, 2)
        parent_b = tournament_select(p.pop, t_size, 2)
        while(parent_a == parent_b):
            parent_b = tournament_select(p.pop, t_size, 2)
       
        child_a, child_b = crossover(parent_a, parent_b)
	if random.random() > 0.55:
	    mutate(child_a)
	if random.random() > 0.95:
	    mutate(child_b)
# EVAL CHILD FITNESS HERE
# when you change impliment truncated replacement
        p.children.extend([child_a, child_b])
        print parent_a.fitness
        print parent_b.fitness
        if parent_a.fitness < parent_b.fitness:
            p.pop.remove(parent_a) #change later
        else:
            p.pop.remove(parent_b)
        print len(p.pop)
    
    p.pop = p.children
    p.children = [] # possible memory errors here.
    epoch = epoch + 1
    print epoch
    print len(p.pop)

evaluate(p.pop, label)
p.pop.sort(key=lambda x: x.fitness, reverse=True)
for i in p.pop:
    print i.fitness
sci.imshow(p.pop[0].img)
#sci.imsave('hr_adversary1.png', c[0].img)


