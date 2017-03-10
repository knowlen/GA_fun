#Author: Nick Knowles (knowlen@wwu.edu)
# Date: Feb 22, 2017
# This is a niave baseline implimentation for a generative machine learning model that 
# approximates a given image using an evolutionary algorithm.
# NOTES: pip installed futures, pip install numpy, pip instal
import scipy.misc as sci
import numpy as np
import random, time, threading
import concurrent.futures
from multiprocessing import Pool
import argparse
import os
from functools import partial
label_shape = []

def get_parser():
    """
    Defines and returns argparse ArgumentParser object.
    :return: ArgumentParser
    """
    parser = argparse.ArgumentParser("Genetic Algorithm for supervised image approximation.")
    parser.add_argument('image_file', type=str, help='The image file for our supervised training.')
    parser.add_argument('results_folder', type=str, help='The folder to print result images.')
    parser.add_argument('-epochs', type=int, default=100, help='Number of iterations to train over')
    parser.add_argument('-mutation_prob', nargs='+', type=float, default=[0.15, 0.05],
                        help='chance for a candidate to be randomly mutated.\nCan assign up to 2 unique values here (one for each child produced in crossover).')
    parser.add_argument("-print_interval",
                        type=int, default=100, 
			help="Prints an image every interval of epochs specified.")
    parser.add_argument('-P', type=int, default=500, help='Population size.')
    parser.add_argument('-ds', type=float, default=80,
                        help='Re-scales image to a resolution of "ds" % the origional (90 = %90 origional size).')
    #parser.add_argument('-debug',action='' ,help='.')  

    return parser




class Semaphore(threading._Semaphore):
    wait = threading._Semaphore.acquire
    signal = threading._Semaphore.release

class Thread(threading.Thread):
    def __init__(self, t, *args):
        threading.Thread.__init__(self, target=t, args=args)
        self.start()

class candidate:
    """
    Defines the candidate object. 
    :Attributes:
        -fitness: how well this candidate performed on evaluation. 
        -img: numpy matrix representation of the image. 
    """
    global label_shape
    def __init__(self, init_state=""):
        self.fitness = 0
        self.rfit = 0
        self.bfit = 0
        self.gfit = 0
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
    """
    Defines the population object.
    :Attributes:
        -pop: A list of randomly initialized candidates. 
        -children: The list of children produced by pop after crossover 
                   and mutation. 
    """
    def __init__(self, size):
    # look at hr sk sample code. 
        self.pop = [candidate() for i in xrange(size)]
        self.children = []
        #for champ in champions:
        #    self.pop.append(champ)
            
    # maybe use if need at some point 
    #def __iter__(self):
    #    return (i for i in (self.pop, self.children))



def evaluate(label, sample):
    """
    Evaluates how close a candidate is to the input image label. 
    Updates a candidate's fitness. 
    """
    #for can in sample:
    
    #error = np.sum(np.absolute(np.subtract(label, sample.img)))
    
        #can.fitness = 1.0/error
        
    error = 0;
    for x,a in zip(sample.img,label):
        for y,b in zip(x,a):
            for z,c in zip(y,b):
                error+= abs(c - z)
    
    sample.fitness = 1.0/error
    return sample

def crossover(p_a, p_b):
    """
    Performs genetic crossover for 2 parent candidates. 
    :Returns: 2 children candidates. 
    """
    mask_a = np.random.choice(2, size=label_shape)
    mask_b = (mask_a - 1) * - 1
    c_a = candidate("blank") 
    c_a.img = (p_a.img * mask_a) + (p_b.img * mask_b)
    c_b = candidate("blank")
    c_b.img = (p_a.img * mask_b) + (p_b.img * mask_a)
    return c_a, c_b
   

def mutate(child):
    """
    Performs a genetic mutation on some child candidate. 
    Updates the child's numpy matrix representation of some image. 
    """
    #look into mutation techniques     
    mask = np.random.choice(2, size=label_shape)
    neg_mask = -np.random.choice(2, size=label_shape)
    scalars = np.random.rand(38,38,4) #* mask) * neg_mask
	#seq element-wise multiplication might be better here
    child.img = child.img + (((child.img*scalars) * mask) * neg_mask)

def tournament_select(pop, t_size, k):
    """
    Selects a candidate psuedo-randomly from the population in a way 
    that preserves diversity, but also favors higher fittness. 
    :Returns: the selected candidate.
    """
    # Tournament Selection
    # Look into applying probability dist if time:
    #   -sort on fitness
    #   -make a new sample_list, give most fit P entries, 2nd most 1-p^2 prob entries, ect..
    selected = candidate()
    for i in xrange(k):
        sample = random.sample(pop, t_size)
        for c in sample:
            if selected.fitness < c.fitness:
                selected = c
            elif selected.fitness == c.fitness:
                selected = random.choice([c, selected]) # might not need the []
    return selected 


def replacement():
    # truncated: take best N from children & pop
    # elitest: take a few best from pop, rest children
    # generational: new p.pop = p.children
    pass


if __name__ == '__main__':
    args = get_parser().parse_args()
    label = sci.imread(args.image_file)
    #label = sci.imread('/home/knowlen/Pictures/goog.png')
    label = sci.imresize(label, args.ds)
    label_shape = label.shape
    t_size = 2 #add to argparse later. Should typically be 2 anyways. 
    p = population(args.P) 
    epoch = args.epochs
    #evaluate(p.pop, label)
    iteration = 0
    #for i in p.pop:
    #    print i.fitness
    img_dir = args.results_folder + '/concurrent__P_' + str(args.P) + '__MaxEpoch_' + str(epoch) 
    os.system('mkdir ' + img_dir) 
    interval = args.print_interval 
    while iteration < epoch: 
        
# Multi processing because Python doesn't support real multi-threading. 
  	pool = Pool(8) 
# partial is needed to pass multiple args to a process function. 
        func = partial(evaluate, label)
    	evaluated = pool.map(func, p.pop)
    	pool.close()
    	pool.join()
	p.pop = evaluated
        #for c in p.pop:
        #	Thread(evaluate, [c], label)
        #evaluate(p.pop, label)
        #with concurrent.futures.ThreadPoolExecutor(max_workers=args.P) as executor: #possible race condition
        #    future_to_url = {executor.submit(evaluate, c, label): c for c in p.pop}
        
        for i in xrange(args.P/2):
            parent_a = tournament_select(p.pop, t_size, 3)
            parent_b = tournament_select(p.pop, t_size, 2)
            while(parent_a == parent_b):
                parent_b = tournament_select(p.pop, t_size, 1)
           
            child_a, child_b = crossover(parent_a, parent_b)
            if random.random() > (1-args.mutation_prob[0]):
                mutate(child_a)
            if random.random() > (1-args.mutation_prob[1]):
                mutate(child_b)
            p.children.extend([child_a, child_b])
            #print parent_a.fitness
            #print parent_b.fitness
            if parent_a.fitness < parent_b.fitness:
                p.pop.remove(parent_a) #change later
            else: #parent_b.fitness < parent_a.fitness:
                p.pop.remove(parent_b)
        
        p.pop = p.children
        p.children = [] # possible memory errors here.
        iteration = iteration + 1
        #print iteration
        if iteration % interval == 0:
            out_fn = img_dir + '/' + str(iteration) + '.png'
            sci.imsave(out_fn, sci.imresize(p.pop[0].img,100))
    
    pool = Pool(8) 
# partial is needed to pass multiple args to a process function. 
    func = partial(evaluate, label)
    evaluated = pool.map(func, p.pop)
    pool.close()
    pool.join()
    p.pop = evaluated
    #evaluate(label, p.pop)
    p.pop.sort(key=lambda x: x.fitness, reverse=True)
   # for i in p.pop:
   #     print i.fitness
#sci.imshow(sci.imresize(p.pop[0].img, 1000))
    out_fn = img_dir + '/final_result.png'
    if not os.path.isdir(args.results_folder):
        os.mkdir(args.results_folder) 
    sci.imsave(out_fn, sci.imresize(p.pop[0].img, 100))

