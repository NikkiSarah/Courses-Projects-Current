import numpy as np

# original ratings matrix
ratings = np.array([[3,3,4,0,4,2,3,0],
                    [3,5,4,3,3,0,0,4],
                    [0,4,0,5,0,0,2,1],
                    [2,0,0,4,0,4,4,5]])
# convert to a boolean matrix where 1 is a rating > 0
purchased = (ratings > 0).astype(int)
# convert to a boolean matrix where 1 is a rating >= 3
good = (ratings > 2).astype(int)

def jaccard_dist(user1, user2):
    dist = (user1 * user2).sum() / ((user1 + user2) > 0).sum()
    return dist

def cosine_dist(user1, user2):
    dist = np.dot(user1, user2) / (np.linalg.norm(user1) * np.linalg.norm(user2))
    return dist

# Question 3
similarity = jaccard_dist(purchased[0], purchased[1])
similarity

# Question 4
similarity = cosine_dist(purchased[2], purchased[3])
similarity

# Question 5
similarity = jaccard_dist(good[0], good[2])
similarity