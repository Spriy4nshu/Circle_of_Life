import numpy as np
from numpy import random
import pandas as pd
from queue import PriorityQueue
        

def graph_creation():
    node_list = {}

    # state defines if that node is empty(-1), has the predator(-10), has the agent(1) or has the prey(10)
    # initially all the noeds are empty
    state = -1

    # making initial connections for each nodes
    for x in range(50):
        connections = []
        if x == 0:
            connect2 = 49
            connections.append(connect2)
        else:
            connect2 = x - 1
            connections.append(connect2)
        
        if x == 49:
            connect1 = 0
            connections.append(connect1)
        else:
            connect1 = x + 1
            connections.append(connect1)

        node_list[x] = {'state' : state, 'connections' : connections, 'degree' : len(connections) }

    y = list(np.arange(50))
    # Making additional connections to each node until the degree is 3 or no other connections are possible
    while y:
        x = random.choice(y)
        x_neighbour = findNeighbor(x)
        z = random.choice(x_neighbour)
        if node_list[z]['degree'] < 3 and node_list[x]['degree'] < 3:
            node_list[z]['degree'] += 1
            node_list[x]['degree'] += 1
            update1 = node_list[x]['connections']
            update1.append(z)
            node_list[x]['connections'] = update1
            update2 = node_list[z]['connections']
            update2.append(x)
            node_list[z]['connections'] = update2
        y.remove(x)

    # making additional list of connections of each nodes
    connection_list = []
    for x in  range(50):
        connection_list.append(node_list[x]['connections'])
    #print(connection_list)

    return node_list, connection_list
    

# finds the neighbours of that node
def findNeighbor(x):
    for_n = []
    for i in range(5):
        element = x + i + 1
        if element == 50:
            element = 0
            for_n.append(element)
        elif element > 50:
            element = element % 49 - 1
            for_n.append(element)
        else:
            for_n.append(element)
    back_n = []
    for j in range(5):
        element = x - (j + 1)
        if element < 0:
            element = 50 - abs(element)
            back_n.append(element)
        else:
            back_n.append(element)

    N_list = back_n[::-1] + for_n
    N_list.remove(N_list[4])
    N_list.remove(N_list[4])
    return N_list


# Finding the distance between 2 nodes using dijkstra algorithm
def dijkstra(start, end, list1):
    count = 0
    last_pos = {}
    g = list(np.zeros(50))
    f = list(np.zeros(50))
    for x in range(50):
        g[x] = float("inf")
    g[start] = 0
    for x in range(50):
        f[x] = float("inf") 
    f[start] = 0
    kyu = PriorityQueue()
    kyu.put((f[start], count, start))
    h_map = {start}
    current = []
    while (len(h_map) > 0):
        current = kyu.get()[2]
        h_map.remove(current)
        node_connections = list1[current]
        for x in node_connections:
            temp_g = g[current] + 1
 
            if (temp_g < g[x]):
                last_pos[x] = current
                g[x] = temp_g
                f[x] = temp_g
                if x not in h_map:
                    count += 1
                    kyu.put((f[x], count, x))
                    h_map.add(x)
    path = []
    last_c = end
    while(last_c in last_pos):
        path.append(last_c)
        last_c = last_pos[last_c]
    
    path = path[::-1]
    return path

# Method that decides predator movement
def predator_movement(predator_position, agent_position, list1):
    predator_neighbour = list1[predator_position]
    path_size = []
    for x in predator_neighbour:
        pred_path = dijkstra(x, agent_position, list1)
        path_size.append(len(pred_path))
    priority_max = min(path_size)
    move_to_list = []
    for y in range(len(path_size)):
        if path_size[y] == priority_max:
            move_to_list.append(predator_neighbour[y])
    move_to = random.choice(move_to_list)
    rand_pos_moves = list1[predator_position]
    rand_pos_move = random.choice(rand_pos_moves)
    new_pos = random.choice([rand_pos_move, move_to], p=[0.4, 0.6])
    return new_pos

# Method that decides predator movement
def prey_movement(prey_position, list1):
    nlist = list1[prey_position]
    new_pos = random.choice(nlist)
    return new_pos

# Finding the node that needs to be survayed using the belief list
def prey_belief_target(belief):
    max_prob = max(belief)
    probable_nodes = []
    for x in range(len(belief)):
        if belief[x] == max_prob:
            probable_nodes.append(x)
    target = random.choice(probable_nodes)
    return target

# updating the belief system of prey
def prey_belief_update(node, clist, belief, prey_pos, case):
    new_belief = belief.copy()
    if case == "agent_move":
        # distribution = belief[node] / (len(belief) - (count + 1))
        for x in range(len(belief)):
            new_belief[x] = belief[x] / (1 - belief[node])
        new_belief[node] = 0

    if case == "drone_check":
        if node == prey_pos:
            for x in range(len(belief)):
                new_belief[x] = 0
            new_belief[node] = 1
        else:
            # distribution = belief[node] / (len(belief) - (count + 1))
            for x in range(len(belief)):
                new_belief[x] = belief[x] / (1 - belief[node])
            new_belief[node] = 0
    
    if case == "prey_move":
        for x in range(len(belief)):
            neighbours = clist[x]
            neighbours_belief = 0
            for neighbour in neighbours:
                neighbours_belief += belief[neighbour] / ((len(clist[neighbour])) + 1)
            new_belief[x] =  (belief[x] / (len(clist[x]) + 1)) + neighbours_belief 
    return new_belief


def agent3(ndict, clist):
    # Randomly spawning agent, predator and prey
    list_of_nodes = list(np.arange(50))
    agent_pos = random.choice(list_of_nodes)
    list_of_nodes.remove(agent_pos)
    ndict[agent_pos]['state'] = 1
    predator_pos = random.choice(list_of_nodes)
    ndict[predator_pos]['state'] = -10
    prey_pos = random.choice(list_of_nodes)
    ndict[prey_pos]['state'] = 10
    belief = list(np.repeat(1/49, 50))
    belief[agent_pos] = 0
    list_of_nodes = list(np.arange(50))
    steps = 0
    win_condition = 'out of steps'
    while steps < 5000:
        probable_prey_pos = prey_belief_target(belief)
        belief = prey_belief_update(probable_prey_pos, clist, belief, prey_pos, "drone_check")
        print(sum(belief))
        # agents move
        ndict[agent_pos]['state'] = -1
        ndict[predator_pos]['state'] = -1
        ndict[prey_pos]['state'] = -1
        curr_dist_prey = dijkstra(agent_pos, probable_prey_pos, clist)
        curr_dist_predator = dijkstra(agent_pos, predator_pos, clist)
        neighbours = clist[agent_pos]
        priority = list(np.zeros(len(neighbours)))
        for x in range(len(neighbours)):
            n_dist_prey = dijkstra(neighbours[x], probable_prey_pos, clist)
            n_dist_predator = dijkstra(neighbours[x], predator_pos, clist)
            if len(curr_dist_prey) > len(n_dist_prey) and len(curr_dist_predator) < len(n_dist_predator):
                priority[x] = 1
            elif len(curr_dist_prey) > len(n_dist_prey) and len(curr_dist_predator) == len(n_dist_predator):
                priority[x] = 2
            elif len(curr_dist_prey) == len(n_dist_prey) and len(curr_dist_predator) < len(n_dist_predator):
                priority[x] = 3
            elif len(curr_dist_prey) == len(n_dist_prey) and len(curr_dist_predator) == len(n_dist_predator):
                priority[x] = 4
            elif len(curr_dist_predator) < len(n_dist_predator):
                priority[x] = 5
            elif len(curr_dist_predator) == len(n_dist_predator):
                priority[x] = 6
            else:
                priority[x] = 7
        priority_max = min(priority)
        move_to_list = []
        for y in range(len(priority)):
            if priority[y] == priority_max:
                move_to_list.append(neighbours[y])
        if priority_max == 7:
            agent_pos = agent_pos
        else:
            agent_pos = random.choice(move_to_list)

        ndict[agent_pos]['state'] = 1
        steps += 1


        if prey_pos == agent_pos:
            win_condition = 'Win'
            break

        if predator_pos == agent_pos:
            win_condition = 'Loss'
            break
        
        belief = prey_belief_update(agent_pos, clist, belief, prey_pos, "agent_move")
        print(sum(belief))
        # predators move
        pred_new_pos = predator_movement(predator_pos, agent_pos, clist)
        predator_pos = pred_new_pos
        ndict[predator_pos]['state'] = -10

        if predator_pos == agent_pos:
            win_condition = 'Loss'
            break

        # preys move
        prey_pos = prey_movement(prey_pos, clist)
        ndict[prey_pos]['state'] = 10
        belief = prey_belief_update(agent_pos, clist, belief, prey_pos, "prey_move")
        print(sum(belief))
        if prey_pos == agent_pos:
            win_condition = 'Win'
            break

    return steps, win_condition, ndict

fields =  ["Graph No", "Win Condition", "Steps"]   
data = pd.DataFrame(columns= fields)
graph = 0
no_of_wins = []
no_of_steps = []
for x in range(30):
    # Generating graphs 30 times
    graph += 1
    ndict, clist = graph_creation()
    wins = 0
    steps1 = 0
    # Running 100 simulations of agnet1 in each of the 30 graphs 
    for y in range(100):
        steps, win_condition, ndict = agent3(ndict, clist)
        steps1 += steps
        if win_condition == 'Win':
            wins += 1
        values = pd.DataFrame([{"Graph No" : graph, "Win Condition" : win_condition, "Steps" : steps}])
        data = pd.concat([data, values], ignore_index=True)
    no_of_wins.append(wins)
    no_of_steps.append(steps1/100)
print(no_of_wins)
print(no_of_steps)
print('done')
data.to_csv('Agent03.csv', index=False)

            
