# Wave Function Collapse implementation based on JavaScript for P5 tutorial from The Coding Train https://www.youtube.com/watch?v=rI_y2GAlQFM
# Wave Function Collapse for 2.5D

import random
import copy


def runWFC(dx, dy, mh, empty):

    #return r3d.Point3dList(points: List[1,2,3]), r3d.Point3dList(points: List[4,5,6]), r3d.Point3dList(points: List[7,8,9])
    #return r3d.Point3d(1,2,3), r3d.Point3d(6,7,8), r3d.Point3d(7,8,9)

    DIM_x = int(dx) #this is i
    DIM_y = int(dy) #this is j
    max_height = int(mh)




    grid = []
    tiles = [] #populated by makeTiles
    listOfTileNames = ['flat', 'sloped', 'corner_up', 'corner_down', 'diagonal', 'NONE_tile'] # used in transfer to GH later to get indices

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

            #Also add empty tile to neighbour possibilities of each tile
            self.up.append(len(tiles)-1) #picking the last tile which is the empty none_tile
            self.down.append(len(tiles)-1)
            self.right.append(len(tiles)-1)
            self.left.append(len(tiles)-1)
                    

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


    def makeTiles(max_height):
        #This creates list of tiles, with rotation. Further down they are copied with heights
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

        #create empty tile
        tiles.append(Tile('NONE_tile', ['N','N','N','N'],0))


        #analyse and populate neighbour possibilities for each Tile
        for each_tile in tiles:
            each_tile.analyse(tiles)

        #populate boundary tile for none_tile
        indices_for_real_tiles = []
        for i in range(len(tiles)-1):
            indices_for_real_tiles.append(i)
        tiles[-1].up = indices_for_real_tiles
        tiles[-1].down = indices_for_real_tiles
        tiles[-1].left = indices_for_real_tiles
        tiles[-1].right = indices_for_real_tiles



    def checkValid(potentialTilesNear, validForCell, direction, oppDir):
        possibleCell = validForCell
        possibleDir = []
        for option in potentialTilesNear:
            # print('this is option counter: ', option)
            possibleDir.append(getattr(tiles[option], oppDir))
        possibleDir = [item for sublist in possibleDir for item in sublist] #flattening list of lists
        return set(possibleCell).intersection(set(possibleDir))


    def cellUpdate():
        updateCount = 0
        for y in range(DIM_y):
            for x in range(DIM_x):
                index = x + y * DIM_x
                if grid[index].collapsed == False:
                    #Look up
                    if y > 0:
                        up = grid[x + (y - 1) * DIM_x]
                        possibleTiles = checkValid(up.tile_options, grid[index].tile_options, 'up', 'down')
                        if possibleTiles != grid[index].tile_options:
                            updateCount += 1
                            grid[index].tile_options = possibleTiles
                    #Look right
                    if x < DIM_x - 1:
                        right = grid[x + 1 + y * DIM_x]
                        possibleTiles = checkValid(right.tile_options, grid[index].tile_options, 'right', 'left')
                        if possibleTiles != grid[index].tile_options:
                            updateCount += 1
                            grid[index].tile_options = possibleTiles
                    #Look down
                    if y < DIM_y - 1:
                        down = grid[x + (y + 1) * DIM_x]
                        possibleTiles = checkValid(down.tile_options, grid[index].tile_options, 'down', 'up')
                        if possibleTiles != grid[index].tile_options:
                            updateCount += 1
                            grid[index].tile_options = possibleTiles
                    #Look left
                    if x > 0:
                        left = grid[x - 1 + y * DIM_x]
                        possibleTiles = checkValid(left.tile_options, grid[index].tile_options, 'left', 'right')
                        if possibleTiles != grid[index].tile_options:
                            updateCount += 1
                            grid[index].tile_options = possibleTiles

                    # Collapse if possible
                    # if len(grid[index].tile_options) == 1:
                    #     grid[index].collapsed == True
        return updateCount
                    


    print('\n')

    print('\nStart code')

    
    makeTiles(max_height)

    #populate grid with all possible tiles
    for i in range(DIM_x*DIM_y):
        if empty[i] == 1: # creates a collapsed empty cell
            grid.append(Cell(
                True,
                [len(tiles)-1] #these are a list of indeces, but we only want the last which is the empty tile
                ))
        elif empty[i] == 0:
            grid.append(Cell(
                False,
                [item for item in range(len(tiles)-1)] #-1 on the end to not count the empty tile at the last index. That way they cannot collapse to empty
                )) #set of indices for possible tiles


    whileloopcount = 0
    while len([item for item in grid if item.collapsed == False]) > 0:
        # print('loop count: ', whileloopcount)
        updateCount = cellUpdate()
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


    '''fake_3d_points = []
    for i in range(len(grid)):
        fake_3d_points.append(r3d.Point3d(listOfTileNames.index(tiles[grid[i].tile_options[0]].name), tiles[grid[i].tile_options[0]].rotation, tiles[grid[i].tile_options[0]].height))
    #print(fake_3d_points)
    #r3d.Point3d()
    return fake_3d_points'''

    #returns name indices, rotation, heights
    return (
    [listOfTileNames.index(tiles[item.tile_options[0]].name) for item in grid],
    [tiles[item.tile_options[0]].rotation for item in grid],
    [tiles[item.tile_options[0]].height for item in grid]
    )



#runWFC(2,2,4, [1,0,0,0])




#take in grid with some that are already collapsed
#make None object with all possible edges?


#restart if no solution found?

