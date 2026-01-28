import time
import numpy as np
from gridgame import *
import random

##############################################################################################################################

# You can visualize what your code is doing by setting the GUI argument in the following line to true.
# The render_delay_sec argument allows you to slow down the animation, to be able to see each step more clearly.

# For your final submission, please set the GUI option to False.

# The gs argument controls the grid size. You should experiment with various sizes to ensure your code generalizes.
# Please do not modify or remove lines 18 and 19.

##############################################################################################################################

game = ShapePlacementGrid(GUI=True, render_delay_sec=0, gs=6, num_colored_boxes=5)
shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute('export')
np.savetxt('initial_grid.txt', grid, fmt="%d")

##############################################################################################################################

# Initialization

# shapePos is the current position of the brush.

# currentShapeIndex is the index of the current brush type being placed (order specified in gridgame.py, and assignment instructions).

# currentColorIndex is the index of the current color being placed (order specified in gridgame.py, and assignment instructions).

# grid represents the current state of the board. 
    
    # -1 indicates an empty cell
    # 0 indicates a cell colored in the first color (indigo by default)
    # 1 indicates a cell colored in the second color (taupe by default)
    # 2 indicates a cell colored in the third color (veridian by default)
    # 3 indicates a cell colored in the fourth color (peach by default)

# placedShapes is a list of shapes that have currently been placed on the board.
    
    # Each shape is represented as a list containing three elements: a) the brush type (number between 0-8), 
    # b) the location of the shape (coordinates of top-left cell of the shape) and c) color of the shape (number between 0-3)

    # For instance [0, (0,0), 2] represents a shape spanning a single cell in the color 2=veridian, placed at the top left cell in the grid.

# done is a Boolean that represents whether coloring constraints are satisfied. Updated by the gridgames.py file.

##############################################################################################################################

shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute('export')

print(shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done)


####################################################
# Timing your code's execution for the leaderboard.
####################################################

start = time.time()  # <- do not modify this.



##########################################
# Write all your code in the area below. 
##########################################

'''
Local Search Algorithm: First choice hill climbing
'''

class Agent:
    def __init__(self, game):
        self.game = game

    def evaluate_state(self, grid, placedShapes):
        score = 0
    
        colored_boxes = (grid != -1).sum()
        score += colored_boxes * 10
    
        violations = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] != -1:
                    if j < len(grid[0]) - 1 and grid[i][j] == grid[i][j + 1]:
                        violations += 1
                    if i < len(grid) - 1 and grid[i][j] == grid[i + 1][j]:
                        violations += 1
    
        score -= violations * 500
        score -= len(placedShapes) * 5
    
        return score

    def generate_random_placement(self):
        # generate a random full move: position, shape, color
        target_shape = random.randint(0, 8)
        target_color = random.randint(0, 3)

        _, _, _, grid, _, _, = self.game.execute("export")
        empty_boxes = np.argwhere(grid == -1)

        if len(empty_boxes) == 0:
            return None

        target_box = empty_boxes[random.randint(0, len(empty_boxes) - 1)]
        target_x, target_y = target_box[1], target_box[0]

        return (target_x, target_y, target_shape, target_color)
    
    def navigate_to(self, target_x, target_y):
        # moves brush to target position
        shapePos, _, _, _, _, _ = self.game.execute("export")
        current_x, current_y = shapePos[0], shapePos[1]

        # move in x direction
        while current_x < target_x:
            self.game.execute("right")
            current_x += 1
        while current_x > target_x:
            self.game.execute("left")
            current_x -= 1

        #move in y direction
        while current_y < target_y:
            self.game.execute("down")
            current_y += 1
        while current_y > target_y:
            self.game.execute("up")
            current_y -= 1

    def switch_to_shape(self, target_shape):
        _, currentShapeIndex, _, _, _, _ = self.game.execute("export")
        while currentShapeIndex != target_shape:
            self.game.execute("switchshape")
            _, currentShapeIndex, _, _, _, _ = self.game.execute("export")

    def switch_to_color(self, target_color): 
        _, _, currentColorIndex, _, _, _ = self.game.execute("export")
        while currentColorIndex != target_color:
            self.game.execute("switchcolor")
            _, _, currentColorIndex, _, _, _ = self.game.execute("export")   

    def solve(self):
        print("Starting solver...")
        max_iterations = 10000
        no_improvement_count = 0

        for iteration in range(max_iterations):
            _, _, _, grid, placedShapes, done = self.game.execute("export")
            if done:
                print(f"solved. Used{len(placedShapes)} shapes")
                return
            
            score_before = self.evaluate_state(grid, placedShapes)

            move = self.generate_random_placement()
            if move is None:
                print("No empty boxes")
                return
            
            target_x, target_y, target_shape, target_color = move

            self.navigate_to(target_x, target_y)
            self.switch_to_shape(target_shape)
            self.switch_to_color(target_color)

            shapePos, currentShapeIndex, _, grid, _, _ = self.game.execute("export")

            if self.game.canPlace(grid, self.game.shapes[currentShapeIndex], shapePos):
                self.game.execute("place")

                _, _, _, new_grid, new_placedShapes, new_done = self.game.execute("export")
                done = new_done

                score_after = self.evaluate_state(new_grid, new_placedShapes)

                if score_after > score_before:
                    no_improvement_count = 0
                else: 
                    self.game.execute("undo")
                    _, _, _, grid, placedShapes, done = self.game.execute("export")
                    no_improvement_count += 1
            else: 
                no_improvement_count += 1
            
            if no_improvement_count >= 500:
                while len(placedShapes) > 0:
                    self.game.execute("undo")
                    _, _, _, grid, placedShapes, done = self.game.execute("export")
                    no_improvement_count = 0
# Run the Agent
print("creating agent...")
solver = Agent(game)
print("calling solve...")
solver.solve()
print("solver finished.")

########################################

# Do not modify any of the code below. 

########################################

end = time.time()

np.savetxt('grid.txt', grid, fmt="%d")
with open("shapes.txt", "w") as outfile:
    outfile.write(str(placedShapes))
with open("time.txt", "w") as outfile:
    outfile.write(str(end - start))