ALPHA = 0.95  # discount factor


class State:
    def __init__(self, key, reward, expected_utility):
        self.reward = reward  # transition reward
        self.key = key  # state
        self.transitions = {}  # transition probabilities
        self.expected_utility = expected_utility  # maximum sum of (discounted) rewards
        self.optimal_action = ""


def states_creation():
    mdp_states = {}
    for i in range(1, 8):  # 7 states
        key = "s" + str(i)
        reward = -0.1  # cost or living reward or transition reward for states s1, s2, s3
        expected_utility = 0

        # terminal states expected utility is same as reward
        if i == 4:  # state 4
            reward = -4
            expected_utility = reward

        if i == 5:  # 5
            reward = 2
            expected_utility = reward

        if i == 6:
            reward = 5
            expected_utility = reward

        if i == 7:
            reward = -2
            expected_utility = reward

        mdp_states[key] = State(key, reward, expected_utility)

    return mdp_states

    # for state in mdp_states:
    #     print(mdp_states[state].key, mdp_states[state].reward, mdp_states[state].expected_utility)


def define_transitions(transitions, action, mdp_states):
    non_terminal_states = []
    for i in range(len(transitions)):  # 0 1 2
        k = "s" + str(i + 1)
        state = mdp_states[k]  # state
        state.transitions[action] = []
        row = transitions[i]
        for c in range(len(row)):
            if row[c] != 0:  # probability != 0
                state_reachable = "s" + str(c + 1)
                state.transitions[action].append((state_reachable, row[c]))

        if action == "walk":
            non_terminal_states.append(state)

    return non_terminal_states


def value_iteration(states, mdp_states):  # value iteration algorithm using bellman equation
    previous_timestep = [0, 0, 0]  # expected utilities for previous timestep

    for i in range(1, 100):
        for state in states:
            current_max = []
            for action in state.transitions:
                exp_utility = state.reward + ALPHA * (sum(
                    [mdp_states[st].expected_utility * prob for st, prob in state.transitions[action]]))
                exp_utility = round(exp_utility, 2)
                if len(current_max) == 0:
                    current_max.append((exp_utility, action))
                elif current_max[0][0] < exp_utility:
                    current_max.pop()
                    current_max.append((exp_utility, action))

            state.expected_utility = current_max[0][0]
            state.optimal_action = current_max[0][1]

        if (abs(previous_timestep[0] - states[0].expected_utility) < 0.01) and (
                abs(previous_timestep[1] - states[1].expected_utility) < 0.01) and (
                abs(previous_timestep[2] - states[2].expected_utility) < 0.01):
            break

        print(f"V{i}(s1):", states[0].expected_utility, f"V{i}(s2):", states[1].expected_utility, f"V{i}(s3):",
              states[2].expected_utility)

        previous_timestep[0] = states[0].expected_utility
        previous_timestep[1] = states[1].expected_utility
        previous_timestep[2] = states[2].expected_utility


if __name__ == '__main__':
    walk = [[0.5, 0.5, 0, 0, 0, 0, 0], [0, 0, 0, 0.5, 0.5, 0, 0], [0, 0, 0.5, 0, 0, 0.5, 0]]
    teleport = [[0, 0.5, 0.5, 0, 0, 0, 0], [0, 0, 0.25, 0.75, 0, 0, 0], [0, 0, 0, 0, 0, 0.5, 0.5]]

    states_mapping = states_creation()
    non_term_states = define_transitions(walk, "walk", states_mapping)
    define_transitions(teleport, "teleport", states_mapping)
    value_iteration(non_term_states, states_mapping)
    for state in non_term_states:
        print(f"{state.key}-utility:", state.expected_utility, "optimal-policy:", state.optimal_action)
