import random
import bisect

def weighted_choice_b(weights):
    totals = []
    running_total = 0
    
    for w in weights:
        running_total += w
        totals.append(running_total)

    rnd = random.random() * running_total
    return bisect.bisect_right(totals, rnd)
    
if __name__ == '__main__':
    x = ((1, 2), (2, 4), (3, 7))
    w = [item[1] for item in x]
    print(weighted_choice_b(w))
