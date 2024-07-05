#%% sample space analysis

# create a sample space of coin flips
sample_space = {'Heads', 'Tails'}

# compute the probability of choosing heads
probability_heads = 1 / len(sample_space)
print(f'Probability of choosing heads is {probability_heads}')

# define event conditions
def is_heads_or_tails(outcome):
    return outcome in {'Heads', 'Tails'}

def is_neither(outcome):
    return not is_heads_or_tails(outcome)

def is_heads(outcome):
    return outcome == "Heads"

def is_tails(outcome):
    return outcome == "Tails"

# define event-detection function
def get_matching_event(event_condition, sample_space):
    return set([outcome for outcome in sample_space if event_condition(outcome)])

event_conditions = [is_heads_or_tails, is_neither, is_heads, is_tails]

for ec in event_conditions:
    print(f"Event condition: {ec.__name__}")
    event = get_matching_event(ec, sample_space)
    print(f"Event: {event}\n")

def compute_prob(event_condition, sample_space):
    event = get_matching_event(event_condition, sample_space)
    return len(event) / len(sample_space)

for ec in event_conditions:
    prob = compute_prob(ec, sample_space)
    name = ec.__name__
    print(f"Probability of event arising from '{name}' is {prob}")
    
# create a weighted sample space (due to a biased coin for example)
weighted_sample_space = {'Heads': 4, 'Tails': 1}
sample_space_size = sum(weighted_sample_space.values())
assert sample_space_size == 5


# redefine the event space
event = get_matching_event(is_heads_or_tails, weighted_sample_space)
event_size = sum(weighted_sample_space[outcome] for outcome in event)
assert event_size == 5

# redine event-detection function
def compute_event_prob(event_condition, sample_space):
    event = get_matching_event(event_condition, sample_space)
    if len(event) == 0:
        return len(event) / len(sample_space)
    else:
        event_size = sum(sample_space[outcome] for outcome in event)
        return event_size / sum(sample_space.values())
    
# output all the event probabilities for a biased coin
for ec in event_conditions:
    prob = compute_event_prob(ec, weighted_sample_space)
    name = ec.__name__
    print(f"Probability of event arising from '{name}' is {prob}")
    
#%% computing non-trivial probabilities

## problem 1: analysing a family with 4 children
# if a family has 4 children, what is the probability that exactly 2 are boys?
# assume each child has an equal probability of being a boy or a girl
from itertools import product

possible_genders = ['Boy', 'Girl']
num_children = 4

# note that product returns a python iterator that can be iterated over only once
all_possible_combinations = product(*(num_children * [possible_genders]))
sample_space = set(all_possible_combinations)

# or even more efficiently
sample_space_efficient = set(product(possible_genders, repeat=num_children))
assert sample_space == sample_space_efficient

# calculate the fraction of sample_space composed of a family with two boys
def has_two_boys(outcome):
    return len([child for child in outcome if child == 'Boy']) == 2

def compute_event_prob2(event_condition, sample_space):
    event = get_matching_event(event_condition, sample_space)
    return len(event) / len(sample_space)

prob = compute_event_prob2(has_two_boys, sample_space)
print(f"Probability of 2 boys is {prob}")
    
## problem 2: analysing multiple dice rolls
# if a 6-sided die is rolled 6 times, what is the probability that the sum adds to exactly
# 21?
# assume the die is fair
possible_outcomes = list(range(1, 7))
print(possible_outcomes)
num_rolls = 6

sample_space = set(product(possible_outcomes, repeat=num_rolls))

def sums_to_21(outcome):
    return sum(outcome) == 21

prob = compute_event_prob2(sums_to_21, sample_space)
print(f"6 rolls sum to 21 with a probability of {round(prob, 4)}")

# using a lambda function
prob = compute_event_prob2(lambda x: sum(x) == 21, sample_space)
assert prob == compute_event_prob2(sums_to_21, sample_space)

## problem 3: computing probabilities with weighted sample spaces
# recompute the probability in problem 2 if the totals possible correspond to how likely
# they are to occur
from collections import defaultdict

weighted_sample_space = defaultdict(int)

for outcome in sample_space:
    total = sum(outcome)
    weighted_sample_space[total] += 1
    
assert weighted_sample_space[6] == 1
assert weighted_sample_space[36] == 1

num_combinations = weighted_sample_space[21]
print(f"There are {num_combinations} ways for 6 rolls of the die to sum to 21")

assert sum([4, 4, 4, 4, 3, 2]) == 21
assert sum([4, 4, 4, 5, 3, 1]) == 21

# there's a direct link between unweighted and weighted event probability computation
# a) the observed count is equal to the length of an unweighted event whose die rolls sum
# to 21
event = get_matching_event(lambda x: sum(x) == 21, sample_space)
assert weighted_sample_space[21] == len(event)
# b) the sum of values in 'weighted_sample' is equal to the length of 'sample_space'
assert sum(weighted_sample_space.values()) == len(sample_space)

# recompute the probability using weighted_sample_space
prob = compute_event_prob(lambda x: x == 21, weighted_sample_space)
assert prob == compute_event_prob2(sums_to_21, sample_space)
print(f"6 rolls sum to 21 with a probability of {round(prob, 4)}")

# the benefit of a weighted sample space over the equivalent unweighted one is lower
# memory usage
print(f"Number of elements in the unweighted sample space: {len(sample_space)}")
print(f"Number of elements in the weighted sample space: {len(weighted_sample_space)}")

#%% computing probabilities over interval ranges
# check whether a number falls within a specific range
def in_interval(number, minimum, maximum):
    return minimum <= number <= maximum

prob = compute_event_prob(lambda x: in_interval(x, 10, 21), weighted_sample_space)
print(f"Probability of interval is {round(prob, 4)}")

## problem 4: what is the probability that 10 coin flips leads to an extreme number of
## heads(i.e. 8 to 10)
# assume the coin is fair

# generate a weighted_sample_space dictionary where the keys are all the number of
# possible heads and the corresponding values are the number of coin flip combinations
# producing that head count
def gen_coin_sample_space(num_flips=10):
    weighted_sample_space = defaultdict(int)
    for flips in product(['Heads', 'Tails'], repeat=num_flips):
        num_heads = len([outcome for outcome in flips if outcome == 'Heads'])
        weighted_sample_space[num_heads] += 1
    return weighted_sample_space

weighted_sample_space = gen_coin_sample_space()
assert weighted_sample_space[10] == 1
assert weighted_sample_space[9] == 10

# calculate the probability of observing 8, 9 or 10 heads
prob = compute_event_prob(lambda x: in_interval(x, 8, 10), weighted_sample_space)
print(f"Probability of observing 8 to 10 heads is {round(prob, 4)}")

## problem 5: what is the probability that 10 coin flips does NOT produce between 3 and 7
## heads?
prob = compute_event_prob(lambda x: not in_interval(x, 3, 7), weighted_sample_space)
print(f"Proabability of observing more than 7 heads or tails is {round(prob, 4)}")
