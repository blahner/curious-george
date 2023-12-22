import os
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque

def prob_N_in_M(N: int, M: int):
    """
    return the probability of getting N consecutive successes in M attempts.
    Using queues, this function uses O(N) space (i.e., the variable N) and runs in O(M) time.
    """
    start = time.time()
    p = 1/32 #1/32 #probability of success. Including 32 characters: 26 letters and special characters " .,?!'"
    q = 1-p #probability of failure
    qpn = q*p**N #just do this computation once and reuse. Dramatic time speedup due to the exponent
    P = deque(maxlen=N+1) #initialize queue. Only need it take up N space
    for _ in range(N): #initialize
        P.append(0)
    P.append(p**N)
    for _ in range(N,M):
        #appending pushes out the last (left-most) value in queue to maintain maxlen
        P.append(P[-1] + (1-P[0])*qpn)
    print(f"time to run code: {time.time() - start} seconds")
    return P[-1]

def until_K_percent_success(N: int, K: float, numMonkeys: int):
    """
    plot the time it takes to get a K% chance of typing N consecutive successes.
    Using queues, this function uses O(N) space (i.e., the variable N)
    """
    start = time.time()
    p = 1/32 #1/32 #probability of success. Including 32 characters: 26 letters and special characters " .,?!'"
    q = 1-p #probability of failure
    qpn = q*p**N #just do this computation once and reuse. Dramatic time speedup due to the exponent
    P = deque(maxlen=N+1) #initialize queue. Only need it take up N space
    for _ in range(N): #initialize
        P.append(0)
    P.append(p**N)
    M=[0] #keep track of probabilities at each "time" for plotting
    x=[0]
    k=0
    count = 0
    while k <= K:
        #appending pushes out the last (left-most) value in queue to maintain maxlen
        P.append(P[-1] + (1-P[0])*qpn)
        count += 1
        k = 1-((1-P[-1])**numMonkeys)
        if (count % 1000000) == 0: #for N=9, sampling every 1000000 is a good number
            M.append(k)
            x.append(count)
    M.append(k) #count the 99% mark
    x.append(count)
    print(f"time to run code: {time.time() - start} seconds")
    return P[-1], M, x

if __name__=='__main__':
    class arguments():
        def __init__(self):
            self.saveroot = os.path.join("/home","blahner","projects","curious-george","output")
            self.numMonkeys = 100000 #how many monkeys are in this experiment? Monkeys work in parallel but don't share results https://www.science.org/content/article/record-number-monkeys-being-used-us-research#:~:text=The%20total%20number%20of%20monkeys,see%20second%20graph%2C%20below)
            self.monkeyLifetime = 40 #time in years a monkey can be expected to live. 40 years in captivity: https://genomics.senescence.info/species/entry.php?species=Macaca_mulatta
            self.cpm = 200 #characters per minute the monkeys can type
    args = arguments()

    N = 14 #there are 14 characters in "Curious George"
    M = int(args.monkeyLifetime*args.cpm*365*24*60) #number of characters each monkey will type in their lifetime

    ## Analysis 1: compute probability of typing sequence of N in the monkeys' lifetime
    print("starting Analysis #1...")
    print(f"{args.numMonkeys} monkeys are typing for {args.monkeyLifetime} years to type a specific sequence of {N} characters...")
    oneMonkeyProb = prob_N_in_M(N, M)
    print(f"Probability of success using just one monkey: {oneMonkeyProb}")
    print(f"Probability of success for at least one of {args.numMonkeys} monkeys: {1-((1-oneMonkeyProb)**args.numMonkeys)}")    
    
    ## Analysis 2: run code until we have a 99% chance that at least one monkey will randomly type a sequence of N in K
    print("starting Analysis #2...")
    K_percent = 0.99
    final_prob, M_tmp, x = until_K_percent_success(N, K_percent, args.numMonkeys)
    print(len(M_tmp))
    x = np.array(x)/(args.cpm*60*24*365) #convert x from character count to time (years)
    print(x[-1])

    #plotting code ...
    plt.plot(x, M_tmp, '-')
    plt.title(f"Probability of a sequence length {N} with {args.numMonkeys} monkeys over time")
    plt.ylim([-0.1,1])
    plt.xlim([0, x[-1]])
    plt.hlines(0,0,x[-1],'k')
    #depending on your parameters, you might want these reference points
    plt.vlines(args.monkeyLifetime,0,1,'r') #40 is lifespan of monkey
    plt.vlines(6000,0,1,'b') #6000 is time since first human civilization
    plt.vlines(350000,0,1,'g') #350000 is time since first human
    plt.vlines(66000000,0,1,'m') #66000000 is time since dinosaurs were wiped out
    plt.vlines(4600000000,0,1,'c') #4,600,000,000 is age of earth
    plt.vlines(13700000000,0,1,'y') #13,700,000,000 is age of universe
    plt.xlabel("Time (years)")
    plt.ylabel("Probability")
    plt.savefig(os.path.join(args.saveroot, f"N-{N}_K-{K_percent}_numMonkeys-{args.numMonkeys}_cpm-{args.cpm}_plot.png"))
    plt.savefig(os.path.join(args.saveroot, f"N-{N}_K-{K_percent}_numMonkeys-{args.numMonkeys}_cpm-{args.cpm}_plot.svg"))
    plt.clf()

    """
    ## Analysis 3: plot the time it takes to get to the 50% mark for increasing values of N
    print("starting Analysis #3...")
    K_percent = 0.5
    N_max = 4
    N_ = []
    for N_tmp in range(N_max):
        final_prob, M_tmp, x = until_K_percent_success(N_tmp, K_percent, args.numMonkeys)
        x = np.array(x)/(args.cpm*60*24*365) #convert x from character count to time (years)
        N_.append(x[-1])

    #plotting code ...
    plt.plot(N_, '-*')
    plt.title(f"{K_percent*100:.0f}% chance at N with {args.numMonkeys} monkeys")
    plt.xlim([0, N_max+1])
    plt.xlabel("N (sequence length)")
    plt.ylabel(f"Time to {K_percent*100:.0f}% (years)")
    plt.savefig(os.path.join(args.saveroot, f"time_to_K-{K_percent}_numMonkeys-{args.numMonkeys}_cpm-{args.cpm}_plot.png"))
    plt.savefig(os.path.join(args.saveroot, f"time_to_K-{K_percent}_numMonkeys-{args.numMonkeys}_cpm-{args.cpm}_plot.svg"))
    plt.clf()
    """
