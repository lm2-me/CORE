# Wave Function Collapse implementation based on JavaScript for P5 tutorial from The Coding Train https://www.youtube.com/watch?v=rI_y2GAlQFM
# Wave Function Collapse for 2.5D

import random
import copy

grid = []
DIM = 10
max_height = 8

class Tile:
    def __init__(self, name, edges, rotation, height=0):
        self.name = name #tile name
        self.edges = edges #edge conditions as list, top, right, btm, left. F Flat, R right-up, L left-up
        self.rotation = rotation #degrees that it is rotated, 0, 90, 180, 270
        self.height = height

        self.up = []
        self.right = []
        self.down = []
        self.left = []
    
            
    def analyse(self, tiles): # new with correct L R opposite pairing because of asymmetry
        for i, tile in enumerate(tiles):
            direction_attr = ['up', 'right', 'down', 'left']
            for dir in range(4):
                #up direction first at index 0 and the tile above index 2. The modulo operater solves the issue of 'list out of range' when dir gets greater than 4.
                if (
                (self.edges[(0+dir)%4] == '0F'     and tile.edges[(2+dir)%4] == '0F' and self.height == tile.height) or
                (self.edges[(0+dir)%4] == '1F'     and tile.edges[(2+dir)%4] == '1F' and self.height == tile.height) or
                (self.edges[(0+dir)%4] == '0F'     and tile.edges[(2+dir)%4] == '1F' and self.height - tile.height == 1) or
                (self.edges[(0+dir)%4] == '1F'     and tile.edges[(2+dir)%4] == '0F' and self.height - tile.height == -1) or
                (self.edges[(0+dir)%4] == 'R'   and tile.edges[(2+dir)%4] == 'L' and self.height == tile.height) or
                (self.edges[(0+dir)%4] == 'L'   and tile.edges[(2+dir)%4] == 'R' and self.height == tile.height) or
                (self.edges[(0+dir)%4] == 'dL'   and tile.edges[(2+dir)%4] == 'R' and self.height - tile.height == 1) or
                (self.edges[(0+dir)%4] == 'dR'   and tile.edges[(2+dir)%4] == 'L' and self.height - tile.height == 1) or
                (self.edges[(0+dir)%4] == 'L'   and tile.edges[(2+dir)%4] == 'dR' and self.height - tile.height == -1) or
                (self.edges[(0+dir)%4] == 'R'   and tile.edges[(2+dir)%4] == 'dL' and self.height - tile.height == -1)
                ):
                    #self.up.append(i)
                    #amend to use getattri
                    edgeMatches = getattr(self, direction_attr[dir])
                    edgeMatches.append(i)
                    setattr(self, direction_attr[dir], edgeMatches)
                
            # tutorial at 48:25
            # possible error correction at 54:52 - fixed

    def rotate(self, angle): #clockwise rotation
        if angle % 90 != 0:
            print('ERROR: rotation angle for tile is not divisible by 90!!!!!! ERROR!!')
        for i in range(int(angle / 90)):
            newEdges = [self.edges[-1]]  + self.edges[:-1]
            self.edges = newEdges
        self.rotation = angle
        return self
    
    def heightMove(self, MoveAmount): #moves tile
        self.height = MoveAmount
        return self




class Cell:
    def __init__(self, collapsed, tile_options):
        self.collapsed = collapsed
        self.tile_options = tile_options    #which index of tiles
        #self.rotation = rotation    #clockwise rotation from base


#This creates list of tiles, with rotation. Further down they are copied with heights
listOfTileNames = ['flat', 'sloped', 'corner_up', 'corner_down', 'diagonal'] # used in transfer to GH later to get indices
tiles = []
tiles.append(Tile('flat', ['0F','0F','0F','0F'], 0))

tiles.append(Tile('sloped', ['1F', 'R', 'F', 'L'], 0))
tiles.append(copy.deepcopy(tiles[1]).rotate(90))
tiles.append(copy.deepcopy(tiles[1]).rotate(180))
tiles.append(copy.deepcopy(tiles[1]).rotate(270))

tiles.append(Tile('corner_up', ['R', '0F', '0F', 'L'], 0))
tiles.append(copy.deepcopy(tiles[5]).rotate(90))
tiles.append(copy.deepcopy(tiles[5]).rotate(180))
tiles.append(copy.deepcopy(tiles[5]).rotate(270))

tiles.append(Tile('corner_down', ['1F', '1F', 'R', 'L'], 0))
tiles.append(copy.deepcopy(tiles[9]).rotate(90))
tiles.append(copy.deepcopy(tiles[9]).rotate(180))
tiles.append(copy.deepcopy(tiles[9]).rotate(270))

tiles.append(Tile('diagonal', ['R', 'dR', 'dL', 'L'], 0))
tiles.append(copy.deepcopy(tiles[13]).rotate(90))
tiles.append(copy.deepcopy(tiles[13]).rotate(180))
tiles.append(copy.deepcopy(tiles[13]).rotate(270))


#copy and create multiple with different heights
for each in range(len(tiles)):
    for i in range(max_height):
        tiles.append(copy.deepcopy(tiles[each]).heightMove(i+1)) # i+1 because height 0 already exists as the original



for each_tile in tiles:
    each_tile.analyse(tiles)


#print('these are the tiles that connect upwards of first ', tiles[14].up)

#populate grid with all possible tiles
for i in range(DIM*DIM):
    grid.append(Cell(
        False,
        [item for item in range(len(tiles))]
        )) #set of indices for possible tiles




for i in range(len(grid)):
    grid[i].tile_options.remove(1)
    grid[i].tile_options.remove(2)
    grid[i].tile_options.remove(3)
    grid[i].tile_options.remove(4)






#print([[item.name, item.edges, item.rotation, item.up] for item in tiles])



# grid[0].tile_options = [1,2]
# grid[3].tile_options = [8,13]

# Pick cell with least entropy



#
#
#print('Printing options before updating')
#print([item.tile_options for item in grid])
#
#
##


# print('printing all edge options')

# for each in tiles:
#     print(each.left)

#print(grid)



# create grid video minute 12 ish


# video at 24:38 and 49:37 for next grid and checkvalid
#make new grid and check and correct tile_optoins. MAke checkValid function.
#checkValid can use two arrays and check against each other.

#just update the original grid
#potentially make loop counting changes/updates, if updates = 0 break loop
#like in K-means
def tileUpdate():
    updateCount = 0
    for j in range(DIM):
        for i in range(DIM):
            index = i + j * DIM
            if grid[index].collapsed == False:
                #Look up
                if j > 0:
                    up = grid[i + (j - 1) * DIM]
                    possibleTiles = checkValid(up.tile_options, grid[index].tile_options, 'up', 'down')
                    if possibleTiles != grid[index].tile_options:
                        updateCount += 1
                        grid[index].tile_options = possibleTiles
                #Look right
                if i < DIM - 1:
                    right = grid[i + 1 + j * DIM]
                    possibleTiles = checkValid(right.tile_options, grid[index].tile_options, 'right', 'left')
                    if possibleTiles != grid[index].tile_options:
                        updateCount += 1
                        grid[index].tile_options = possibleTiles
                #Look down
                if j < DIM - 1:
                    down = grid[i + (j + 1) * DIM]
                    possibleTiles = checkValid(down.tile_options, grid[index].tile_options, 'down', 'up')
                    if possibleTiles != grid[index].tile_options:
                        updateCount += 1
                        grid[index].tile_options = possibleTiles
                #Look left
                if i > 0:
                    left = grid[i - 1 + j * DIM]
                    possibleTiles = checkValid(left.tile_options, grid[index].tile_options, 'left', 'right')
                    if possibleTiles != grid[index].tile_options:
                        updateCount += 1
                        grid[index].tile_options = possibleTiles

                # Collapse if possible
                # if len(grid[index].tile_options) == 1:
                #     grid[index].collapsed == True
    return updateCount
                

def checkValid(potentialTilesNear, validForCell, direction, oppDir):
    possibleCell = validForCell
    possibleDir = []
    for option in potentialTilesNear:
        possibleDir.append(getattr(tiles[option], oppDir))
    possibleDir = [item for sublist in possibleDir for item in sublist] #flattening list of lists
    return set(possibleCell).intersection(set(possibleDir))



print('\n')

print('\nStart code')

whileloopcount = 0
while len([item for item in grid if item.collapsed == False]) > 0:
    # print('loop count: ', whileloopcount)
    updateCount = tileUpdate()
    # print('update counter: ', updateCount)

    if updateCount == 0:
        gridCopy = grid
        gridCopy = sorted(gridCopy, key= lambda e:len(e.tile_options))
        gridCopy = [item for item in gridCopy if item.collapsed == False] # remove collapsed ones
        lowest_entropy = [item for item in gridCopy if len(gridCopy[0].tile_options) == len(item.tile_options)]
        #lambda gridCopy: gridCopy[0].tile_options == gridCopy[i].tile_options, lowest_entropy.append(gridCopy[i])
        #print('print lowest entropy: ', list(lowest_entropy))
        # print('These are the tile options', [item.tile_options for item in lowest_entropy])
        #now pick the ones with lowest entropy and select one randomly to randomly collapse
        if len(lowest_entropy) > 0:
            picked = random.choice(lowest_entropy)
            picked.collapsed = True
            picked.tile_options = [random.choice(tuple(picked.tile_options))]
            # print('this is picked ', picked.collapsed, picked.tile_options)
        
    whileloopcount += 1




print([tiles[item.tile_options[0]].name for item in grid])

print('rotations')
print([tiles[item.tile_options[0]].rotation for item in grid])

print('name indices')
print([listOfTileNames.index(tiles[item.tile_options[0]].name) for item in grid])

print('heights')
print([tiles[item.tile_options[0]].height for item in grid])







#take in grid with some that are already collapsed
#make None object with all possible edges?


#restart if no solution found
