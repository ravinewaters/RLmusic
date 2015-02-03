__author__ = 'redhat'

# preprocess.py
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