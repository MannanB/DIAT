import numpy as np
import random
import pygame


class CommEnv:
    """
    A grid environment with two obstacle types (triangle, circle) and survivors.

    - The 'speaker' can see the entire grid, but only triangle obstacles (not circles).
    - The 'listener' can see only its 4-directional neighborhood plus its own cell,
      can see circle obstacles but NOT triangle obstacles.
    - Survivors are denoted by 'X'.
    - The agent (listener) is denoted by 'A' in the grid (for internal use / rendering).
    """

    # Cell type annotations for internal representation
    EMPTY = 0
    CIRCLE_OBS = -1 # we wont spawn these for now
    TRIANGLE_OBS = 1
    SURVIVOR = 2
    AGENT = 3  # optional to store agent in the grid for rendering

    def __init__(
        self,
        grid_size=8,
        num_circle_obstacles=3,
        num_triangle_obstacles=1,
        num_survivors=1,
        max_timesteps=None,
        render_mode=None
    ):
        """
        Initialize the CommEnv.

        :param grid_size: Size of the grid (n x n).
        :param num_circle_obstacles: Number of circle obstacles to place randomly.
        :param num_triangle_obstacles: Number of triangle obstacles to place randomly.
        :param num_survivors: Number of survivors to place randomly.
        :param render_mode: If 'human', attempts pygame rendering; otherwise None.
        """
        self.grid_size = grid_size
        self.num_circle_obstacles = num_circle_obstacles
        self.num_triangle_obstacles = num_triangle_obstacles
        self.num_survivors = num_survivors

        self.max_timesteps = max_timesteps
        self.steps = 0

        self.render_mode = render_mode
        self.window = None
        self.cell_size = 40  # for pygame rendering

        # Actions: 0=Up, 1=Right, 2=Down, 3=Left, 4=Stay
        self.action_space = [0, 1, 2, 3]#, 4]

        self.grid = None
        self.agent_pos = (0, 0)
        self.reset()

        if self.render_mode == 'human':
            self._init_pygame()

    def _init_pygame(self):
        """Initialize pygame if needed."""
        pygame.init()
        self.window = pygame.display.set_mode(
            (self.grid_size * self.cell_size, self.grid_size * self.cell_size)
        )
        pygame.display.set_caption("CommEnv")

    def _quit_pygame(self):
        """Quit pygame if it was initialized."""
        if self.window is not None:
            pygame.quit()

    def reset(self, locations=None):
        # Create empty grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # self._place_random_objects(self.num_circle_obstacles, self.CIRCLE_OBS) <-- we wont spawn these for now
        if locations is None:
            self._place_random_objects(self.num_triangle_obstacles, self.TRIANGLE_OBS)
            self._place_random_objects(self.num_survivors, self.SURVIVOR)
        else:
            self.grid[locations[0]] = self.TRIANGLE_OBS
            self.grid[locations[1]] = self.SURVIVOR

        self.agent_pos = self._get_free_cell()
        self.grid[self.agent_pos] = self.AGENT

        self.steps = 0

        return self.get_speaker_obs(), self.get_listener_obs()

    def step(self, action):
        # Compute the intended new position
        dx, dy = 0, 0
        if action == 0:   # Up
            dx, dy = -1, 0
        elif action == 1: # Right
            dx, dy = 0, 1
        elif action == 2: # Down
            dx, dy = 1, 0
        elif action == 3: # Left
            dx, dy = 0, -1

        new_x = self.agent_pos[0] + dx
        new_y = self.agent_pos[1] + dy

        reward = 0.0
        done = False

        # Check boundary and obstacle collision

        
        # reward for getting closer to A survivor (there can be 1 or 2)
        survivor_pos = np.where(self.grid == self.SURVIVOR)
        if len(survivor_pos[0]) == 1:
            survivor_x, survivor_y = survivor_pos[0][0], survivor_pos[1][0]
            current_dist = abs(survivor_x - self.agent_pos[0]) + abs(survivor_y - self.agent_pos[1])
            new_dist = abs(survivor_x - new_x) + abs(survivor_y - new_y)
            if new_dist < current_dist:
                reward += 2
            else:
                reward -= 4
        elif len(survivor_pos[0]) == 2:
            survivor1_x, survivor1_y = survivor_pos[0][0], survivor_pos[1][0]
            survivor2_x, survivor2_y = survivor_pos[0][1], survivor_pos[1][1]
            current_dist1 = abs(survivor1_x - self.agent_pos[0]) + abs(survivor1_y - self.agent_pos[1])
            current_dist2 = abs(survivor2_x - self.agent_pos[0]) + abs(survivor2_y - self.agent_pos[1])
            new_dist1 = abs(survivor1_x - new_x) + abs(survivor1_y - new_y)
            new_dist2 = abs(survivor2_x - new_x) + abs(survivor2_y - new_y)
            new_dist = min(new_dist1, new_dist2)
            current_dist = min(current_dist1, current_dist2)
            if new_dist < current_dist:
                reward += 0.5
            else:
                reward -= 0.5


        if self._valid_move(new_x, new_y):
            if self._collided_with_obstacle(new_x, new_y):
                done = True
                reward = -5.0
            elif self._collided_with_survivor(new_x, new_y):
                reward = 10.0
            # Remove agent from old position
            self.grid[self.agent_pos] = self.EMPTY
            # Update agent position
            self.agent_pos = (new_x, new_y)
            # Place agent on new position
            self.grid[self.agent_pos] = self.AGENT
        else:
            reward = -1.0

        # check if all survivors are rescued
        if np.sum(self.grid == self.SURVIVOR) == 0:
            done = True
            reward = 15.0

        if self.max_timesteps is not None and self.steps >= self.max_timesteps:
            done = True
            reward = -10

        # Return new observations
        speaker_obs = self.get_speaker_obs()
        listener_obs = self.get_listener_obs()

        # discourage reward hacking
        if reward == 0:
            reward = -0.01

        self.steps += 1

        return speaker_obs, listener_obs, reward, done

    def invalid_actions(self):
        # find invalid actions
        invalid_actions = []
        for a in self.action_space:
            dx, dy = 0, 0
            if a == 0:
                dx, dy = -1, 0
            elif a == 1:
                dx, dy = 0, 1
            elif a == 2:
                dx, dy = 1, 0
            elif a == 3:
                dx, dy = 0, -1

            new_x = self.agent_pos[0] + dx
            new_y = self.agent_pos[1] + dy
            if not (self._valid_move(new_x, new_y) and not self._collided_with_obstacle(new_x, new_y)):
                invalid_actions.append(a)

        return invalid_actions

    def render(self, wait=False):
        """
        Render the environment using pygame (if render_mode='human' and pygame is installed).
        """
        if self.render_mode != 'human':
            return

        # Pump pygame events (needed to avoid "freezing" on some systems)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.window.fill((255, 255, 255))  # white background

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell_type = self.grid[r, c]
                rect = pygame.Rect(
                    c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size
                )
                if cell_type == self.EMPTY:
                    # Draw empty cell
                    pygame.draw.rect(self.window, (200, 200, 200), rect, 1)
                elif cell_type == self.CIRCLE_OBS:
                    # Draw circle obstacle (filled circle in cell)
                    pygame.draw.rect(self.window, (200, 200, 200), rect)
                    center = (
                        c * self.cell_size + self.cell_size // 2,
                        r * self.cell_size + self.cell_size // 2
                    )
                    pygame.draw.circle(self.window, (0, 0, 255), center, self.cell_size // 3)
                elif cell_type == self.TRIANGLE_OBS:
                    # Draw triangle obstacle
                    pygame.draw.rect(self.window, (200, 200, 200), rect)
                    p1 = (c * self.cell_size + self.cell_size // 2, r * self.cell_size + 5)
                    p2 = (c * self.cell_size + 5, r * self.cell_size + self.cell_size - 5)
                    p3 = (c * self.cell_size + self.cell_size - 5, r * self.cell_size + self.cell_size - 5)
                    pygame.draw.polygon(self.window, (255, 0, 0), [p1, p2, p3])
                elif cell_type == self.SURVIVOR:
                    # Draw survivor (X)
                    pygame.draw.rect(self.window, (200, 200, 200), rect)
                    font = pygame.font.SysFont(None, 24)
                    text_surface = font.render('X', True, (0, 128, 0))
                    text_rect = text_surface.get_rect(
                        center=(
                            c * self.cell_size + self.cell_size // 2,
                            r * self.cell_size + self.cell_size // 2
                        )
                    )
                    self.window.blit(text_surface, text_rect)
                elif cell_type == self.AGENT:
                    # Draw agent
                    pygame.draw.rect(self.window, (255, 255, 0), rect)

        pygame.display.flip()
        if wait:
        # wait until n is unpressed
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.KEYUP:
                        if event.key == pygame.K_n:
                            return
                    if event.type == pygame.QUIT:
                        pygame.quit()
        pygame.time.delay(100)

    def get_speaker_obs(self):
        # speaker_grid = np.zeros_like(self.grid)

        # for r in range(self.grid_size):
        #     for c in range(self.grid_size):
        #         val = self.grid[r, c]
        #         if val == self.TRIANGLE_OBS:
        #             speaker_grid[r, c] = self.TRIANGLE_OBS
        #         elif val == self.SURVIVOR:
        #             speaker_grid[r, c] = self.SURVIVOR
        #         else:
        #             # Hide circle obstacles or anything else
        #             speaker_grid[r, c] = self.EMPTY

        # get position of agent
        agent_pos = np.where(self.grid == self.AGENT)
        agent_x, agent_y = agent_pos[0][0], agent_pos[1][0]
        single_num = agent_x * self.grid_size + agent_y
        return [single_num]#(agent_x, agent_y)

    def get_listener_obs(self):
        # Positions for current/up/right/down/left
        obs = []

        for rx in range(self.grid_size):
            for ry in range(self.grid_size):
                if 0 <= rx < self.grid_size and 0 <= ry < self.grid_size:
                    cell_val = self.grid[rx, ry]
                    if cell_val == self.AGENT:
                        # Hide triangle obstacles
                        obs.append(self.EMPTY)
                    else:
                        # Keep circle obstacles, survivors as is
                        obs.append(cell_val)
                else:
                    # Out of bounds is considered empty (or could be a wall if you prefer)
                    obs.append(self.EMPTY)

        return np.array(obs, dtype=int).flatten()
    
    def count_survivors(self):
        return np.sum(self.grid == self.SURVIVOR)

    def _valid_move(self, x, y):
        return not (x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size)

    def _collided_with_obstacle(self, x, y):
        cell_val = self.grid[x, y]
        return cell_val == self.CIRCLE_OBS or cell_val == self.TRIANGLE_OBS
        
    def _collided_with_survivor(self, x, y):
        cell_val = self.grid[x, y]
        return cell_val == self.SURVIVOR

    def _place_random_objects(self, count, cell_type):
        """Place 'count' objects of given cell_type in random free cells."""
        for _ in range(count):
            free_cell = self._get_free_cell()
            self.grid[free_cell] = cell_type

    def _get_free_cell(self):
        """Return a random free cell (row, col) in the grid."""
        free_cells = [(r, c) for r in range(self.grid_size)
                      for c in range(self.grid_size)
                      if self.grid[r, c] == self.EMPTY]
        if not free_cells:
            raise ValueError("No free cells left to place an object.")
        return random.choice(free_cells)


if __name__ == "__main__":
    env = CommEnv(grid_size=8, render_mode='human')  # or None if you don't want to visualize
    speaker_obs, listener_obs = env.reset()

    for _ in range(50):
        # run random rollout
        env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = random.choice(env.action_space)
            _, _, reward, done = env.step(action)
            total_reward += reward
            env.render()
        
        print("Total reward:", total_reward)
    
    # Close pygame if needed
    if env.render_mode == 'human':
        pygame.quit()