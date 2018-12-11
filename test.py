import numpy as np
import random
import math
import time
from simanneal import Annealer

# Padding around each table in meters
w_padding = 1.4
h_padding = 1.4

# number of table
n = 12
# Table safe area size
w = 1.5 + w_padding
h = 1.05 + h_padding

# Size of each room
# {room code : [w, h]}
# room 1 is Active Team Space
# room 2 is Active Team Space North
rooms = {1: [19, 16],
         2: [19, 10]}
state = []


# Rotate vector v at original with degree r clockwise starting at point (x, y)
def rotation(x, y, v, r):
    theta = np.radians(30 * r)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, s), (-s, c)))
    # print (np.round(R.dot(v), 4))
    return np.round(R.dot(v), 4) + np.array([x, y])


def overlap(xmin, xmax, ymin, ymax):
    for each in state:
        if xmax >= each.xmin and each.xmax >= xmin and ymax >= each.ymin and each.ymax >= ymin:
            return True
    return False


class Rec:
    # Initialize a rectangle area by given width, height
    # original position at (x , y) of the bottom left point
    def __init__(self, w, h, x, y, r, room_code):
        self.w = w
        self.h = h
        self.x = x
        self.y = y
        self.r = r
        self.room_code = room_code
        a = np.array([0, h])
        b = np.array([w, 0])
        self.v1 = np.array([x, y])
        self.v2 = rotation(self.x, self.y, a, self.r)
        self.v4 = rotation(self.x, self.y, b, self.r)
        self.v3 = rotation(self.v4[0], self.v4[1], a, self.r)
        self.xmin = self.v1[0]
        self.xmax = self.v3[0]
        self.ymin = self.v4[1]
        self.ymax = self.v2[1]
    # Verify if each rectangle is in constraint area

    def valid(self):
        # Vertex other than (x, y) and index order is clockwise
        # W = rooms[self.room_code][0]
        # H = rooms[self.room_code][1]
        W = 19
        H = 10
        if self.v1[0] < 0 or self.v1[0] > W or self.v2[0] < 0 or self.v2[0] > W or self.v3[0] < 0 or self.v3[0] > W or self.v4[0] < 0 or self.v4[0] > W:
            # print(1, v2, v3, v4, self.r)
            return False
        if self.v1[1] < 0 or self.v1[1] > H or self.v2[1] < 0 or self.v2[1] > H or self.v3[1] < 0 or self.v3[1] > H or self.v4[1] < 0 or self.v4[1] > H:
            # print(2, v2, v3, v4)
            return False

        # Make sure no overlap of each Rec
        # for eachRec in state:
        if overlap(self.xmin, self.xmax, self.ymin, self.ymax):
            return False
        return True

windows = {
    1: [[1.37, 0, 0], [4.07, 0, 0], [6.78, 0, 0], [9.5, 0, 0], [12.21, 0, 0], [14.93, 0, 0]],
    2: []
}

# Hard-code locations of outlets for each room.
outlets = {
    1: [[0, 14.7, 0], [19, 14.7, 0], [2.03, 0, 0], [3.39, 0, 0], [4.75, 0, 0], [6.11, 0, 0], [7.46, 0, 0], [8.82, 0, 0], [10.18, 0, 0],
        [11.5, 0, 0], [12.9, 0, 0], [14.25, 0, 0], [15.6, 0, 0], [17.0, 0, 0]],
    2: [[9.5, 9.5, 0], [6.78, 9.5, 0], [0, 4.3, 0], [19, 4.3, 0]]
}

# For simulated annealing, we want to minimize some cost function.
# In the context of this problem, we want to minimize the unpleasantness of a floor layout.


def distance(p0, p1):
    return math.sqrt((p0.x - p1[0])**2 + (p0.y - p1[1])**2)


def overall_unpleasantness_score():
    # Cost function we want to minimize is the total unpleasantness score
    # of all tables in room 1 and (something about the
    # number of tables is related, not sure if minusing the number of tables is the correct thing to do here -- consider
    # norming all values?)

    return total_table_unpleasantness(2)
    # return total_table_unpleasantness(state, 1) - len(state)


def pairwisedis():
    s = 0
    for i in range(n):
        for j in range(i+1, n):
            s = s + distance(state[i], np.array([state[j].x, state[j].y]))

    return s


def total_table_unpleasantness(room_code):
    total_unpleasantness = 0
    dist_window = 0
    dist_outlet = 0
    for table in state:
        if windows[room_code]:
            dist_window = find_dist_to_closest_window(table, room_code)
        if outlets[room_code]:
            dist_outlet = find_dist_to_closest_outlet(table, room_code)
        total_unpleasantness += 10 * dist_window + 20 * dist_outlet - 15 * pairwisedis()
    return total_unpleasantness


def find_dist_to_closest_window(table, room_code):
    min_window_dist = distance(table, windows[room_code][0])
    for window in windows[room_code]:
        dist = distance(table, window)
        if dist < min_window_dist:
            min_window_dist = dist
    return min_window_dist


def find_dist_to_closest_outlet(table, room_code):
    min_outlet_dist = distance(table, outlets[room_code][0])
    for outlet in outlets[room_code]:
        dist = distance(table, outlet)
        if dist < min_outlet_dist:
            min_outlet_dist = dist
    return min_outlet_dist

class TableArrange(Annealer):

    def __init__(self, state):
        super(TableArrange, self).__init__(state)

    def anneal(self):
        """Minimizes the energy of a system by simulated annealing.
        Parameters
        state : an initial arrangement of the system
        Returns
        (state, energy): the best state and energy found.
        """
        step = 0
        self.start = time.time()

        # Precompute factor for exponential cooling from Tmax to Tmin
        if self.Tmin <= 0.0:
            raise Exception('Exponential cooling requires a minimum "\
                "temperature greater than zero.')
        Tfactor = -math.log(self.Tmax / self.Tmin)

        # Note initial state
        T = self.Tmax
        E = self.energy()
        prevState = self.copy_state(self.state)
        prevEnergy = E
        self.best_state = self.copy_state(self.state)
        self.best_energy = E
        trials, accepts, improves = 0, 0, 0
        if self.updates > 0:
            updateWavelength = self.steps / self.updates
            self.update(step, T, E, None, None)

        # Attempt moves to new states
        while step < self.steps and not self.user_exit:
            step += 1
            T = self.Tmax * math.exp(Tfactor * step / self.steps)
            self.move(T)
            E = self.energy()
            dE = E - prevEnergy
            trials += 1
            if dE > 0.0 and math.exp(-dE / T) < random.random():
                # Restore previous state
                self.state = self.copy_state(prevState)
                E = prevEnergy
            else:
                # Accept new state and compare to best state
                accepts += 1
                if dE < 0.0:
                    improves += 1
                prevState = self.copy_state(self.state)
                prevEnergy = E
                if E < self.best_energy:
                    self.best_state = self.copy_state(self.state)
                    self.best_energy = E
            if self.updates > 1:
                if (step // updateWavelength) > ((step - 1) // updateWavelength):
                    self.update(
                        step, T, E, accepts / trials, improves / trials)
                    trials, accepts, improves = 0, 0, 0
        self.state = self.copy_state(self.best_state)
        if self.save_state_on_exit:
            self.save_state()

        # Return best state and energy
        return self.best_state, self.best_energy

    def move(self, T):
        print(T)
        change = random.randint(0, n - 1)
        old = state.pop(change)
        dx = np.random.normal(0, T/8000)
        dy = np.random.normal(0, T/6000)
        newRec = Rec(w, h, old.v1[0] + dx, old.v1[1] + dy, 0, 1)
        while not (newRec.valid()):
            dx = np.random.normal(0, T / 5000)
            dy = np.random.normal(0, T / 12500)
            newRec = Rec(w, h, old.v1[0] + dx, old.v1[1] + dy, 0, 1)
        state.append(newRec)

    def energy(self):
        return overall_unpleasantness_score()

# Initialize random floorplan
for i in range(n):
    randomRec = Rec(w, h, random.randint(0, 19), random.randint(0, 10), 0, 1)
    while not randomRec.valid():
        randomRec = Rec(w, h, random.randint(0, 19), random.randint(0, 10), 0, 1)
    state.append(randomRec)
f = TableArrange(state)
f.steps = 50000
state, e = f.anneal()
for table in state:
    print([table.x, table.y, table.r],",")


















