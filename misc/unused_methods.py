__author__ = 'redhat'

# run.py
# def make_dict_of_elem(list_of_tuples):
#     """
#     Assume that we have a list of tuples.
#     An element in n-th coordinate of a tuple comes from a set.
#     The set contains all elements that are in n-th coordinate of all
#     tuples in the list.
#     Map all the elements from the set to a nonnegative integer and vice versa.
#     Returns 3 lists.
#     The first list contains dictionaries of k-th element to nonnegative integer
#     The second is the other way around.
#     the third contains size of each set
#     """
#
#     # initialize list of dictionaries
#     list_to_int = []
#     list_to_elem = []
#     list_size = []
#
#     # assume that all tuples have same length
#     for i in range(len(list_of_tuples[0])):
#         list_to_int.append({})  # key: elem, value: int
#         list_to_elem.append({})  # key: int, value: elem
#         counter = 0
#         for tuple in list_of_tuples:
#             # unpack elements in the tuple
#             # store it in dictionary if the key doesn't exist, pass otherwise
#             if tuple[i] not in list_to_int[i]:
#                 list_to_int[i][tuple[i]] = counter
#                 list_to_elem[i][counter] = tuple[i]
#                 counter += 1
#         list_size.append(counter)
#     return (list_to_int, list_to_elem, list_size)
#
# def convert_elem_of_states_to_int(list_of_lists_of_tuples):
#     """
#     returns states that are tuple of nonnegative integers, e.g.
#     (1,1,1,1,2) instead of ((0, 1.0), 'C', 1.0, 1.0, 0)
#     """
#
#     # flatten list of list of tuples to list of tuples
#     flatten_states = [item for list_of_tuples in
#                       list_of_lists_of_tuples for item in list_of_tuples]
#
#     global dict_elem_to_int, dict_int_to_elem, elem_set_sizes
#     dict_elem_to_int, dict_int_to_elem, elem_set_sizes = \
#         make_dict_of_elem(flatten_states)
#
#     new_all_states = []
#     for states in list_of_lists_of_tuples:
#         new_states = []
#         for state in states:
#             new_state = []
#             for i in range(len(state)):
#                 new_elem = dict_elem_to_int[i][state[i]]
#                 new_state.append(new_elem)
#             new_states.append(tuple(new_state))
#         new_all_states.append(new_states)
#     return new_all_states

def make_action_by_duration_dict(list_of_song_states, actions_dict):
    # need to convert action to int first using map_tuples_to_int
    # then separate each action according to its duration
    flat_states = make_flat_list(list_of_song_states)
    action_set_by_duration = {}
    for state in flat_states:
        if state[1] != 'pickup':
            if state[2] in action_set_by_duration:
                    action_set_by_duration[state[2]].append(
                        actions_dict[0][state[:2]])
            else:
                action_set_by_duration[state[2]] = [
                    actions_dict[0][state[:2]]]
    for key in action_set_by_duration:
        action_set_by_duration[key] = set(action_set_by_duration[key])
    return action_set_by_duration

def map_tuples_to_int(flat_states):
    """
    return a 2-tuple of dict of item to int and int to item
    """
    tuple_to_int_dict = {}
    int_to_tuples_dict = {}
    counter = 0
    for state in flat_states:
        if state not in tuple_to_int_dict:
            tuple_to_int_dict[state] = counter
            int_to_tuples_dict[counter] = state
            counter += 1
    return tuple_to_int_dict, int_to_tuples_dict

def map_item_inside_list_of_list(list_of_lists, item_mapper):
    # item_mapper is a dict
    mapped_list_of_list = []
    for lists in list_of_lists:
        mapped_item_list = []
        for item in lists:
            int_state = item_mapper[item]
            mapped_item_list.append(int_state)
        mapped_list_of_list.append(mapped_item_list)
    return mapped_list_of_list

def map_state_action_pair(states_dict, actions_dict):
    state_action_to_int_dict = {}
    int_to_state_action_dict = {}
    state_action_sizes = (len(states_dict[0]), len(actions_dict[0]))
    for int_s in states_dict[1]:
        for int_a in actions_dict[1]:
            if is_valid_action(states_dict[1][int_s], actions_dict[1][int_a]):
                integer = array_to_int((int_s, int_a), state_action_sizes)
                state_action_to_int_dict[int_s, int_a] = integer
                int_to_state_action_dict[integer] = (int_s, int_a)
    return state_action_to_int_dict, int_to_state_action_dict

def get_features_range(all_states, all_actions, terminal_states):
    for state in all_states:
        if state in terminal_states:
            continue
        for action in all_actions:
            if not is_valid_action(state, action):
                continue

            features_vector = compute_features(state, action, terminal_states)

            try:
                min_features_vector = np.minimum(min_features_vector,
                                                 features_vector)
                max_features_vector = np.maximum(max_features_vector,
                                                 features_vector)
            except NameError:
                min_features_vector = features_vector
                max_features_vector = features_vector

    return min_features_vector, max_features_vector
# These methods are assumed to have input the original state and action not
# integers.

def compute_features_matrix(elem_sizes, fignotes_dict, chords_dict,
                             terminal_states):
    state_action_sizes = (
        len(states_dict[0]), len(actions_dict[0]))  # (# states, # actions)
    min_feat, max_feat = get_features_range(states_dict, actions_dict,
                                            terminal_states)
    num_of_rows = state_action_sizes[0] * state_action_sizes[1]

    # Use DOK sparse matrix

    first = True
    for state in states_dict[0]:
        if state in terminal_states:
            continue
        for action in actions_dict[0]:
            if not is_valid_action(state, action):
                continue
            int_s = states_dict[0][state]
            int_a = actions_dict[0][action]
            features_vector = compute_features(state, action, terminal_states)

            # row = array_to_int((int_s, int_a), state_action_sizes)
            row = state_action_dict[0][int_s, int_a]
            if first:
                # when first iteration, initialize sparse matrix after
                # having computer number of columns of features vector
                col, num_of_cols = map_tup_to_bin_array(features_vector,
                                                        min_feat,
                                                        max_feat)
                sparse_feature_matrix = sparse.dok_matrix((num_of_rows,
                                                           num_of_cols),
                                                          dtype=np.uint8)
                first = False
            else:
                col, _ = map_tup_to_bin_array(features_vector, min_feat,
                                              max_feat)
            for j in col:
                sparse_feature_matrix[row, j] = 1

    return sparse_feature_matrix.tocsr()



def map_tup_to_bin_array(tup, min_elem, max_elem):
    """
    Goal: turns a tuple into a binary array.
    Input: a tuple of integers, the min (and max) value each coordinate
    can have.
    Output: array of indices that correspond to value 1 in binary tuple of
    the original tuple. Also output the length of the binary tuple

    """
    coord_size = np.array(max_elem) - np.array(min_elem) + 1
    bin_array_length = sum(coord_size)
    coord_size = np.concatenate((np.array([0]), coord_size))
    pos = []
    index = 0
    for i in range(len(tup)):
        index = index + coord_size[i]
        pos.append(index + tup[i] - min_elem[i])
    return np.array(pos), bin_array_length

def generate_possible_action(state_action_dict):
    possible_action = {}
    for state_action in state_action_dict[0]:
        if state_action[0] in possible_action:
            possible_action[state_action[0]].append(state_action[1])
        else:
            possible_action[state_action[0]] = [state_action[1]]
    return possible_action


def choose_action_from_policy_matrix(policy_matrix, all_actions,
                          state, state_size, action_size):
    # doesn't check whether state is a terminal state
    reduced_state = state[:3]
    int_s = array_to_int(reduced_state[::-1], state_size[::-1])
    indices = policy_matrix[int_s].indices
    row = policy_matrix[int_s].data
    prob = row/sum(row)
    int_a = np.random.choice(indices, p=prob)  # int
    key_a = tuple(int_to_array(int_a, action_size[::-1])[::-1])
    # print(key_a)
    action = key_a + all_actions[key_a]
    return action


def generate_trajectory_based_on_errors(state, term_states, q_states,
                                        errors, gamma):
    # original state
    # all_actions dict
    trajectory = []
    while True:
        trajectory.append(state)
        if state in errors:  # if the row has nonzero entries.

            # with prob. gamma, choose random action
            if random() < gamma:
                # q_states[state] is a dictionary
                action = choice(list(q_states[state]))
            # with prob. 1-gamma, based on errors. Smaller error,
            # lesser chance to be chosen.
            else:
                # simulate probability
                idx = weighted_choice_b(errors[state].values())
                actions = list(errors[state])
                try:
                    action = actions[idx]
                except IndexError:
                    action = actions[-1]
        else:
            action = choice(list(q_states[state]))
        trajectory.append(action)

        if state in term_states and action == -1:
            break

        state = compute_next_state(state, action)
    return trajectory


def dict_argmax(dict):
    return max(dict.items(), key=lambda x: x[1])[0]



def array_to_int(arr, elem_size):
    arr = np.array(arr)
    sizes = np.array((1,) + elem_size[:-1])
    cum_prod = np.cumprod(sizes)
    return np.dot(arr, cum_prod)


def int_to_array(integer, elem_size):
    sizes = np.array((1,) + elem_size[:-1])
    cum_prod = np.cumprod(sizes)
    index = -1
    arr = [0]*len(elem_size)
    for radix in reversed(cum_prod):
        q, integer = divmod(integer, radix)
        arr[index] = q
        index -= 1
    return arr

def generate_all_states(new_list_of_song_states):
    # combine all figures with all beat, but subjected to restriction
    # return list of all_states
    flatten_states = make_flat_list(new_list_of_song_states)
    figure = []
    for item in flatten_states:
        figure.append(item[:3] + item[-2:],)

    figure = set(figure)
    beat = list(range(2, 20, 2))

    all_states = []
    for item in itertools.product(figure, beat):
        duration = item[0][3]
        beat = item[1]
        if beat + duration <= 20:
            key = item[0][:3] + (item[1],)
            if key not in all_states:
                value = (item[0][-2:])
                all_states.append(value)

    save_obj(all_states, 'ALL_STATES')
    return all_states



def weighted_choice(choices):
    choices = tuple(choices)
    total = sum(w for c, w in choices)
    r = random() * total
    upto = 0
    for c, w in choices:
      if upto + w >= r:
         return c
      upto += w
    assert False, "Shouldn't get here"


def is_valid_action(state, action):
    # valid action iff
    # get_current_beat + duration + action duration <= 20
    fig_duration = state[4]
    fig_beat = state[3]
    action_fig_duration = action[2]
    if fig_beat + fig_duration < 20:
        if action_fig_duration + fig_beat + fig_duration <= 20:
            return True
        return False # if > 20, false
    elif fig_beat + fig_duration == 20:
        return True
    assert False, "Shouldn't get here"


    def q_learning(self, reward_mtx, disc_rate, max_reward, n_iter=50):
        # q-learning
        # use for loop over all actions. The size of states and actions is not
        # too large
        # max_values = {s : (q_value, (a,)}
        # q_states = {s : {a: (row_idx, s')}}
        # q_matrix = {(state, action): [q-value, n_visit]}

        q_states = self.q_states
        q_matrix = {}
        max_values = dict.fromkeys(list(q_states), (0, None))
        for _ in itertools.repeat(None, n_iter):
            for state, actions in q_states.items():
                for action in actions:
                    row_idx = q_states[state][action][0]
                    reward = reward_mtx[row_idx]
                    if action != -1:
                        state_prime = q_states[state][action][1]
                        sample = reward + disc_rate * max_values[state_prime][0]
                    else:
                        sample = reward + max_reward

                    if (state, action) not in q_matrix:
                        q_matrix[(state, action)] = [5, 1]
                        alpha = 0
                    else:
                        alpha = 1/q_matrix[(state, action)][1]
                    new_q_value = alpha * sample + \
                                  (1-alpha) * q_matrix[(state, action)][0]

                    q_matrix[(state, action)][0] = new_q_value
                    q_matrix[(state, action)][1] += 1

                    # update max_values
                    if max_values[state][0] <= new_q_value:
                        max_values[state] = (new_q_value, (action,))
        policy_matrix = {s: v[1] for s, v in max_values.items()}
        return policy_matrix