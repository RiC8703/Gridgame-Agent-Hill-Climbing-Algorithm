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

game = ShapePlacementGrid(GUI=True, render_delay_sec=0.2, gs=6, num_colored_boxes=5)
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

This implementation uses first choice hill climbing to solve the grid coloring problem.
At each iteration, it generates a random move (position, shape, color) and evaluates the resulting state.
If the new state has a better score, then the move is accepted; otherwise, it is undone.
Restarts are performed if no improvement is found after a certain number of no_improvement_count.
'''

class Agent:
    def __init__(self, game):
        # Intialize the agent with the game instance.
        self.game = game

    def evaluate_state(self, grid, placedShapes):
        """
        Objective function to evaluate the current state of grid.
        Return a score based on:
        - Number of colored boxes (higher is better)
        - Number of violations (lower is better)
        - Number of shapes used (lower is better)

        """
        score = 0

        # reward colored boxes
        colored_boxes = (grid != -1).sum()
        score += colored_boxes * 10

        # count violations (adjacent boxes of same color)
        violations = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] != -1:
                    if j < len(grid[0]) - 1 and grid[i][j] == grid[i][j + 1]:
                        violations += 1
                    if i < len(grid) - 1 and grid[i][j] == grid[i + 1][j]:
                        violations += 1

        # penalize violations and number of shapes used
        score -= violations * 500
        score -= len(placedShapes) * 5
    
        return score

    def get_available_colors_set(self, grid, x, y, shape_cells=None):
        """
        Returns set of colors that won't create adjacency violation at (x, y). 
        (x,y) = (col, row).
        shape_cells: set of (col, row) tuples that are part of the shape being placed (to ignore).
        """
        gs = len(grid)
        adjacent_colors = set()
        # Check left neighbor
        if x > 0:
            nx, ny = x - 1, y
            if (shape_cells is None or (nx, ny) not in shape_cells) and grid[ny, nx] != -1:
                adjacent_colors.add(grid[ny, nx])
        # Check right neighbor
        if x < gs - 1:
            nx, ny = x + 1, y
            if (shape_cells is None or (nx, ny) not in shape_cells) and grid[ny, nx] != -1:
                adjacent_colors.add(grid[ny, nx])
        # Check top neighbor
        if y > 0:
            nx, ny = x, y - 1
            if (shape_cells is None or (nx, ny) not in shape_cells) and grid[ny, nx] != -1:
                adjacent_colors.add(grid[ny, nx])
        # Check bottom neighbor
        if y < gs - 1:
            nx, ny = x, y + 1
            if (shape_cells is None or (nx, ny) not in shape_cells) and grid[ny, nx] != -1:
                adjacent_colors.add(grid[ny, nx])
        return set(range(4)) - adjacent_colors

    def get_valid_colors_for_placement(self, grid, shape, pos):
        """
        Returns intersection of available colors for all cells the shape would cover. 
        pos = [x, y]. Excludes cells that are part of the shape itself from neighbor checks.
        """
        # First, collect all cells that will be filled by this shape
        shape_cells = set()
        for i in range(len(shape)):
            for j in range(len(shape[0])):
                if shape[i, j]:
                    cx, cy = pos[0] + j, pos[1] + i
                    shape_cells.add((cx, cy))
        
        # Now check valid colors for each cell, excluding shape cells from neighbor checks
        valid = None
        for i in range(len(shape)):
            for j in range(len(shape[0])):
                if shape[i, j]:
                    cx, cy = pos[0] + j, pos[1] + i
                    avail = self.get_available_colors_set(grid, cx, cy, shape_cells)
                    valid = avail if valid is None else (valid & avail)
        return valid if valid is not None else set(range(4))

    def generate_random_placement(self):
        # get current grid state and find empty boxes
        _, _, _, grid, _, _ = self.game.execute("export")
        empty_boxes = np.argwhere(grid == -1)

        if len(empty_boxes) == 0:
            return None

        # Prefer positions adjacent to filled cells for better coverage
        # Score each empty position by number of filled neighbors
        position_scores = []
        for box in empty_boxes:
            y, x = int(box[0]), int(box[1])
            score = 0
            gs = len(grid)
            # Check neighbors
            if x > 0 and grid[y, x - 1] != -1:
                score += 1
            if x < gs - 1 and grid[y, x + 1] != -1:
                score += 1
            if y > 0 and grid[y - 1, x] != -1:
                score += 1
            if y < gs - 1 and grid[y + 1, x] != -1:
                score += 1
            position_scores.append((score, (y, x)))
        
        # Weight selection: prefer positions with more neighbors (but still allow some randomness)
        if random.random() < 0.7:  # 70% chance to prefer positions with neighbors
            position_scores.sort(reverse=True)
            # Pick from top 30% of positions
            top_n = max(1, len(position_scores) // 3)
            target_y, target_x = position_scores[random.randint(0, top_n - 1)][1]
        else:
            # Random selection
            target_box = empty_boxes[random.randint(0, len(empty_boxes) - 1)]
            target_y, target_x = int(target_box[0]), int(target_box[1])

        # bias toward larger shapes (fewer shapes = higher score): weight by num cells squared
        # Squaring gives even more preference to larger shapes
        shape_weights = [np.sum(s) ** 2 for s in self.game.shapes]
        total = sum(shape_weights)
        r = random.uniform(0, total)
        for target_shape in range(9):
            r -= shape_weights[target_shape]
            if r <= 0:
                break

        # pick color that avoids violations for this shape at this position
        pos = [target_x, target_y]
        if self.game.canPlace(grid, self.game.shapes[target_shape], pos):
            valid_colors = self.get_valid_colors_for_placement(grid, self.game.shapes[target_shape], pos)
            if valid_colors:
                target_color = random.choice(list(valid_colors))
            else:
                # No valid color exists - try a different shape/position
                return None
        else:
            # Shape doesn't fit - try a different shape/position
            return None

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
        # switch brush to target shape
        _, currentShapeIndex, _, _, _, _ = self.game.execute("export")
        while currentShapeIndex != target_shape:
            self.game.execute("switchshape")
            _, currentShapeIndex, _, _, _, _ = self.game.execute("export")

    def switch_to_color(self, target_color): 
        # switch brush to target color
        _, _, currentColorIndex, _, _, _ = self.game.execute("export")
        while currentColorIndex != target_color:
            self.game.execute("switchcolor")
            _, _, currentColorIndex, _, _, _ = self.game.execute("export")   

    def solve(self):
        """
        Implementation of First Choice Hill Climbing with Random Restarts.
    
        Algorithm:
        1. generate a random move. (position, shape, color)
        2. navigate to position, switch to shape and color.
        3. place the shape if possible.
        4. evaluate the new state.
        5. if the new state score is better, keep the move or else undo it.
        6. if no improvement after certain tries, restart from start.

        """
        print("Starting solver...")
        max_iterations = 100000  # more iterations to find better solutions
        no_improvement_count = 0  # counter for no improvements
        best_score = float('-inf')
        best_shape_count = float('inf')

        for iteration in range(max_iterations):
            _, _, _, grid, placedShapes, done = self.game.execute("export")
            if done:
                print(f"Solved! Used {len(placedShapes)} shapes")
                return
            
            # evaluate current state
            score_before = self.evaluate_state(grid, placedShapes)

            # generate random move (try multiple times if needed to find valid placement)
            move = None
            for attempt in range(200):  # try up to 200 times to find a valid move
                move = self.generate_random_placement()
                if move is not None:
                    break
            
            if move is None:
                no_improvement_count += 1
                if no_improvement_count >= 400:
                    # restart if stuck
                    while len(placedShapes) > 0:
                        self.game.execute("undo")
                        _, _, _, grid, placedShapes, done = self.game.execute("export")
                    no_improvement_count = 0
                continue
            
            target_x, target_y, target_shape, target_color = move

            # execute the move
            self.navigate_to(target_x, target_y)
            self.switch_to_shape(target_shape)
            self.switch_to_color(target_color)

            # get updated position after navigation
            shapePos, currentShapeIndex, _, grid, _, _ = self.game.execute("export")

            # try placing the shape
            if self.game.canPlace(grid, self.game.shapes[currentShapeIndex], shapePos):
                self.game.execute("place")

                # get and evaluate new state
                _, _, _, new_grid, new_placedShapes, new_done = self.game.execute("export")
                done = new_done
                
                # Check if placement created violations (even if not complete)
                has_violations = False
                for i in range(len(new_grid)):
                    for j in range(len(new_grid[0])):
                        if new_grid[i][j] != -1:
                            if j < len(new_grid[0]) - 1 and new_grid[i][j] == new_grid[i][j + 1]:
                                has_violations = True
                                break
                            if i < len(new_grid) - 1 and new_grid[i][j] == new_grid[i + 1][j]:
                                has_violations = True
                                break
                    if has_violations:
                        break
                
                if has_violations:
                    # Violation created - undo immediately
                    self.game.execute("undo")
                    _, _, _, grid, placedShapes, done = self.game.execute("export")
                    no_improvement_count += 1
                else:
                    score_after = self.evaluate_state(new_grid, new_placedShapes)
                    # first choice hill climbing: keep only if improvement
                    if score_after > score_before:
                        no_improvement_count = 0
                        # Track best solution
                        if len(new_placedShapes) < best_shape_count:
                            best_shape_count = len(new_placedShapes)
                            best_score = score_after
                    else: 
                        self.game.execute("undo")
                        _, _, _, grid, placedShapes, done = self.game.execute("export")
                        no_improvement_count += 1
            else: 
                no_improvement_count += 1

            # random restart if stuck (restart sooner to escape local optima)
            if no_improvement_count >= 400:
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

# get final state to check if done
shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute("export")
print("Final state:", done)
print("Final Grid:\n", grid)

########################################

# Do not modify any of the code below. 

########################################

end = time.time()

np.savetxt('grid.txt', grid, fmt="%d")
with open("shapes.txt", "w") as outfile:
    outfile.write(str(placedShapes))
with open("time.txt", "w") as outfile:
    outfile.write(str(end - start))