import numpy as np
from rlberry.envs.finite import FiniteMDP
from rlberry.rendering   import RenderInterface2D, Scene, GeometricPrimitive


class Chain(FiniteMDP, RenderInterface2D):
    """
    Simple chain environment. 
    Reward 0.05 in initial state, reward 1.0 in final state.
    """
    def __init__(self, L=5, fail_prob=0.1):
        """
        Parameters
        ----------
        L : int 
            length of the chain
        fail_prob : double 
            fail probability 
        """
        assert L >= 2
        self.L = L

        # transition probabilities
        P = np.zeros((L, 2, L))
        for ss in range(L):
            for aa in range(2):
                if ss == 0:
                    P[ss, 0, ss]   = 1.0-fail_prob  # action 0 = don't move
                    P[ss, 1, ss+1] = 1.0-fail_prob  # action 1 = right
                    P[ss, 0, ss+1] = fail_prob  
                    P[ss, 1, ss]   = fail_prob          
                elif ss == L-1:
                    P[ss, 0, ss-1] = 1.0-fail_prob  # action 0 = left
                    P[ss, 1, ss]   = 1.0-fail_prob  # action 1 = don't move
                    P[ss, 0, ss]   = fail_prob  
                    P[ss, 1, ss-1] = fail_prob 
                else:
                    P[ss, 0, ss-1] = 1.0-fail_prob  # action 0 = left
                    P[ss, 1, ss+1] = 1.0-fail_prob  # action 1 = right
                    P[ss, 0, ss+1] = fail_prob  
                    P[ss, 1, ss-1] = fail_prob 
        
        # mean reward
        S = L
        A = 2 
        R = np.zeros((S, A))
        R[L-1, :] = 1.0
        R[0,   :] = 0.05

        # init base classes
        FiniteMDP.__init__(self, R, P, initial_state_distribution=0)
        RenderInterface2D.__init__(self)
        self.id = "Chain"
        self.reward_range = (0.0, 1.0)

        # rendering info
        self.set_clipping_area((0, L, 0, 1))
        self.set_refresh_interval(500)  # in milliseconds


    def step(self, action):
        assert action in self._actions, "Invalid action!"

        # save state for rendering
        if self.is_render_enabled():
            self.append_state_for_rendering(self.state)
        
        # take step
        next_state, reward, done, info = self.sample(action, self.state)
 
        self.state = next_state
        return next_state, reward, done, info


    #
    # Code for rendering
    #

    def get_background(self):
        """
        Returne a scene (list of shapes) representing the background
        """
        bg = Scene()
        colors = [ (0.8, 0.8, 0.8),  (0.9, 0.9, 0.9) ]
        for ii in range(self.L):
            shape = GeometricPrimitive("GL_QUADS")
            shape.add_vertex((ii,   0))
            shape.add_vertex((ii+1, 0))
            shape.add_vertex((ii+1, 1))
            shape.add_vertex((ii,   1))
            shape.set_color(colors[ii%2])
            bg.add_shape(shape)
    

        flag = GeometricPrimitive("GL_TRIANGLES")
        flag.set_color((0.0, 0.5 ,0.0))
        x = self.L - 0.5 
        y = 0.25
        flag.add_vertex((x, y))
        flag.add_vertex((x+0.25, y+0.5))
        flag.add_vertex((x-0.25, y+0.5))
        bg.add_shape(flag)


        return bg 

    def get_scene(self, state):
        """
        Return scene (list of shapes) representing a given state
        """
        scene = Scene()

        agent = GeometricPrimitive("GL_QUADS")
        agent.set_color((0.75, 0.0, 0.5))

        size = 0.25
        x = state + 0.5
        y = 0.5

        agent.add_vertex((x-size/4.0, y-size))
        agent.add_vertex((x+size/4.0, y-size))
        agent.add_vertex((x+size/4.0, y+size))
        agent.add_vertex((x-size/4.0, y+size))

        agent.add_vertex((x-size, y-size/4.0))
        agent.add_vertex((x+size, y-size/4.0))
        agent.add_vertex((x+size, y+size/4.0))
        agent.add_vertex((x-size, y+size/4.0))

        scene.add_shape(agent)
        return scene  
    