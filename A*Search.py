# Finish A* search function that can find path from starting point to the end
# The robot starts from start position (0,0) and finds a path to end position (4, 5)
# In the maze, 0 is open path while 1 means wall (a robot cannot pass through wall)
# heuristic is provided

# example result:
# [[0, -1, -1, -1, -1, -1],
#  [1, -1, -1, -1, -1, -1],
#  [2, -1, -1, -1, -1, -1],
#  [3, -1,  8, 10, 12, 14],
#  [4,  5,  6,  7, -1, 15]]

maze = [[0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0]]

heuristic = [[9, 8, 7, 6, 5, 4],
             [8, 7, 6, 5, 4, 3],
             [7, 6, 5, 4, 3, 2],
             [6, 5, 4, 3, 2, 1],
             [5, 4, 3, 2, 1, 0]]

start = [0, 0] # starting position
end = [len(maze)-1, len(maze[0])-1] # ending position
cost = 1 # cost per movement

move = [[-1, 0 ], # go up
         [ 0, -1], # go left
         [ 1, 0 ], # go down
         [ 0, 1 ]] # go right


### finish the A* search funciton below
def search(maze, start, end, cost, heuristic):
    from operator import add

    nrows = len(maze)
    ncols = len(maze[0])

    # This function evalues if a given position is an Obstacle
    def isObstacle(position):
        x= position[0]
        y=position[1]
        res= False
        if (x < 0 or x >= nrows or y < 0 or y >= ncols): res = True
        else: res = (maze[x][y]  == 1)
        return res
    #This function returns valid neighbors
    def getNeighbors(position):
        res=[]
        for x in move:
            neighbor =list( map(add, position, x) )
            if not isObstacle(neighbor):
                res.append(neighbor)
        return res
    #The variable res initialise the path in the maze, all positions are set to -1 
    res = [[-1 for _ in range(ncols)] for _ in range(nrows)] 
    res[start[0]][start[1]] = 0
    visited = [] #storing all position visited
    cost = 0
    position = start
    res[position[0]][position[1]] = cost 
    visited.append(position) #list of positions already visited
    while position != end:
        #Find valid neighbors (not obstacles)
        neighbors = getNeighbors(position)
        print(position)
        print(cost)
        #remove neighbors that will require to go back to current position 
        for x in neighbors:  
            if len(getNeighbors(x)) == 1 : 
                # if only 1 neighbor and not the end, it's a dead path
                if x!=end: neighbors.remove(x) 
        #remove neighbors that have been visited
        for x in neighbors:
            if x in visited: neighbors.remove(x)  
        if len(neighbors) == 1:
            position = neighbors[0]
        if len(neighbors)> 1:
            # select the neighbor with lowest cost from heuristic 
            h = [heuristic[x[0]][x[1]] for x in neighbors]
            position = neighbors[h.index(min(h))]
        cost +=1
        res[position[0]][position[1]] = cost
        visited.append(position)
    return res
