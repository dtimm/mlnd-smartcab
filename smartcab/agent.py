import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        # Initialize any additional variables here
        self.q_states = {}
        self.alpha = 0.9
        self.trials = 0
        self.successes = 0
        self.reward = 0
        self.total_reward = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # Prepare for a new trip; reset any variables here, if required
        # 
        self.trials += 1

        #print self.q_states
        self.total_reward += self.reward
        
        # decay alpha
        self.alpha *= 0.8

        self.reward = 0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        self.state = (inputs['light'],self.next_waypoint,inputs['oncoming'],inputs['left'],inputs['right']) 
        
        # Select action according to your policy
        action = self.get_action(self.state)

        if self.state in self.q_states.keys():
            if action == None:
                expected_reward = self.q_states[self.state]['none']
            else:
                expected_reward = self.q_states[self.state][action]
        else:
            expected_reward = 1.0

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Check if we made it to the finish.
        if self.planner.next_waypoint() == None:
            # We made it!
            self.successes += 1
            print '{0}/{1}'.format(self.successes, self.trials)
            print 'running average reward: {0:.1f}'.format(self.total_reward/self.trials)

        # Learn policy based on state, action, reward
        # Run the update function.
        self.q_func(self.state, action, reward)#, s_prime)

        self.reward += reward
        print 'state: {}, deadline = {}, action = {}, reward = {}, expected = {}'.format(self.state, deadline, action, reward, expected_reward)
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def get_action(self, s):
        # Given a state, return an action.

        # duplicate the simple case as a base if you haven't dealt with traffic yet.
        found = s in self.q_states.keys()
        if not found:
            copy_state = s[:2] + (None, None, None)
            if copy_state in self.q_states.keys():
                found = True
                self.q_states[s] = self.q_states[copy_state]

        if found:
            best = -10
            # default action is random if nothing is better.
            best_act = random.choice([None, 'left', 'right', 'forward'])
            for k, v in self.q_states[s].iteritems():
                if v > best:
                    best = v
                    best_act = k

            if best_act == 'none':
                action = None
            else:
                action = best_act
        else:
            # 
            # take a random actions if you've never felt like this before.
            action = random.choice([None, 'left', 'right', 'forward'])
            self.q_states[s] = {'none':3, 'left':3, 'right':3, 'forward':3}

        return action

    def q_func(self, s, a, r): 
        if a == None:
            a = 'none'

        self.q_states[s][a] = (1.0 - self.alpha) * self.q_states[s][a] + self.alpha * r

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
