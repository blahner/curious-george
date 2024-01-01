# curious-george
This code was used in the writing of my Medium post, ["Curious George Discovers the Infinite Monkey Theorem"](tmp).

This code explores the [Infinite Monkey Theorem](https://en.wikipedia.org/wiki/Infinite_monkey_theorem) by simulating monkey(s) randomly typing on a keyboard. At its core, the code calculates the probability of randomly achieving a sequence N successes in K attempts (see my [previous post](https://medium.com/@benlahner/what-is-the-probability-of-flipping-n-consecutive-heads-in-k-coin-flips-71fcae35c33a) on an explanation of this probability). 

I code three different analyses that might be of interest:
1. compute probability of typing sequence of N successes in the monkeys' lifetime (or any time period).
2. run code until we have a 99% chance that at least one monkey will randomly type a sequence of N successes in K attempts
3. compute and plot the time it takes to get to the 50% probability mark for increasing values of N. Additionally fit an exponential function to extrapolate to long sequences.
