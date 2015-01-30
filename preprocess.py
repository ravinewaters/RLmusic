__author__ = 'redhat'

from music21 import converter, note, harmony, stream
import os
import pickle
from pprint import pprint

# Method name should start with a verb

def parse(filename):
    """
    Idea:
    1. When it's a chord, the next note is a figurehead
    2. Continue acquiring all notes that falls into the same figure, until
    3. meeting the next chord, which implies new figure
    4. Pickup bar and the last figure are handled separately
    """
    states = []
    song = converter.parse(filename)
    elements = song.flat.getElementsByClass([harmony.ChordSymbol, note.Rest,
                                     note.Note])

    anacrusis = False
    first_measure = song.parts[0][1]

    # check anacrusis here instead of inside loop
    if first_measure.duration != first_measure.barDuration:
        anacrusis = True

        pickup = []
        pickup_fighead = None


        for elem in first_measure.notesAndRests:
            if elem.isNote:
                if not pickup_fighead:
                    pickup_fighead = elem.midi
                    pickup_beat = elem.beat
                    pickup_duration = 0
                pickup.append(elem.midi)

            elif elem.isRest and pickup_fighead:
                pickup.append('rest')

            if pickup_fighead:
                pickup.append(elem.quarterLength)
                pickup_duration += elem.quarterLength

        states.append((tuple(pickup), "pickup", pickup_duration,
                       pickup_beat, pickup_fighead))

    last_item = False
    for i in range(len(elements)):
        prev_elem = elements[i-1]
        elem = elements[i]
        try:
            next_elem = elements[i+1]
        except stream.StreamException:
            # at the last iteration, look for next item in the original measure
            next_elem = elem.next()
            last_item = True

        # Skip if anacrusis
        if elem.measureNumber == 1 and anacrusis:
            continue

        if elem.isChord:
            # get chord's name
            fig_chord = elem.figure

        elif elem.isNote:
            if prev_elem.isChord or elem.beat == 1.0:
                fighead = elem.midi
                fig_start_at_beat = elem.beat
                fig_notes = []
                fig_duration = 0

            fig_notes.append(elem.midi)
            fig_notes.append(elem.quarterLength)
            fig_duration += elem.quarterLength

            # Wrap up figure if we encounter Rest or Chord or new bar (beat ==
            # 1.0) or Final Barline
            if not hasattr(next_elem, 'pitch') or next_elem.beat == 1.0:
                figure = (tuple(fig_notes),
                          fig_chord,
                          fig_duration,
                          fig_start_at_beat,
                          fighead)

                states.append(figure)

        elif elem.isRest and not last_item:
            # elem is a rest
            states.append((('rest', elem.quarterLength), 'rest', elem.quarterLength, elem.beat,
                           'rest'))

    return states

def get_corpus(corpus_dir):
    filenames = []
    for f in os.listdir(corpus_dir):
        if '.xml' in f and os.path.isfile(corpus_dir + f):
            filenames.append(corpus_dir + f)
    return filenames

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

def make_flat_list(list_of_lists):
    flat_list = [item for lists in list_of_lists for item in lists]
    return flat_list

def make_list_of_all_action(list_of_song_states):
    flat_states = make_flat_list(list_of_song_states)
    all_actions = []
    for state in flat_states:
        if state[1] != 'pickup':
            all_actions.append(state[:2])
    return all_actions

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

def get_terminal_states(trajectories):
    terminal_states = []
    for trajectory in trajectories:
        terminal_states.append(trajectory[-1])
    return set(terminal_states)

def get_start_states(trajectories):
    start_states = []
    for trajectory in trajectories:
        start_states.append(trajectory[0])
    return set(start_states)


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


def compute_action(int_s_prime, states_dict, actions_dict):
    # states_dict: (states_to_int, int_to_states)
    # actions_dict: (actions_to_int, int_to_actions)
    # find a that makes fs transition to s_prime

    s_prime = states_dict[1][int_s_prime]
    a = (s_prime[0], s_prime[1])
    return actions_dict[0][a]

def get_trajectories(list_of_song_states, states_dict, actions_dict):
    trajectories = []
    for song_states in list_of_song_states:
        trajectory = []

        first = True
        for int_s in song_states:
            if first:
                trajectory.append(int_s)
                first = False
                continue
            int_a = compute_action(int_s, states_dict, actions_dict)
            trajectory.append(int_a)
            trajectory.append(int_s)
        trajectories.append(trajectory)
    return trajectories

def compute_next_state(int_s, int_a, states_dict, actions_dict):
    # states_dict: (states_to_int, int_to_states)
    # actions_dict: (actions_to_int, int_to_actions)
    # a = ((next_fig_seq_of_notes), next_chord_name)
    # s = ((fig_seq_of_notes), chord_name, duration, beat, fighead_note)
    # s_prime = ((next_fig_seq_of_notes),
    # next_chord_name,
    # duration = sum of duration of each note in next_fig_seq_of_notes,
    # beat = duration of s + beat s in,
    # fighead_note = the first note of next_fig_seq_of_notes)

    s = states_dict[1][int_s]
    a = actions_dict[1][int_a]
    s_prime = (a[0], a[1], sum(a[0][1::2]), s[2]+s[3], a[0][0])
    return states_dict[0][s_prime]

def make_dir_when_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def save_obj(obj, name):
    dir = 'obj/'
    make_dir_when_not_exist(dir)
    with open(dir + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    dir = 'obj/'
    make_dir_when_not_exist(dir)
    with open(dir + name + '.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    filenames = get_corpus('corpus/')
    list_of_song_states = []
    for filename in filenames:
        states = parse(filename)
        list_of_song_states.append(states)

    # TRAJECTORIES is list of trajectory
    # Trajectory is (s1, a1, s2, a2, s3, a3, ...)

    # IMPORTANT data:
    # states_dict, actions_dict and trajectories


    all_actions = make_list_of_all_action(list_of_song_states)
    # print("\n ALL ACTIONS")
    # pprint(all_actions)

    actions_dict = map_tuples_to_int(all_actions)
    print("\n ACTIONS DICT")
    pprint(actions_dict)
    save_obj(actions_dict, 'ACTIONS_DICT')

    actions_by_duration_dict = make_action_by_duration_dict(list_of_song_states,
                                                          actions_dict)
    print("\nACTIONS by DURATION DICT")
    pprint(actions_by_duration_dict)
    save_obj(actions_by_duration_dict, 'ACTIONS_BY_DURATION_DICT')

    states_dict = map_tuples_to_int(make_flat_list(list_of_song_states))
    print("\nSTATES DICT")
    pprint(states_dict)
    save_obj(states_dict, 'STATES_DICT')

    new_list_of_song_states = map_item_inside_list_of_list(list_of_song_states,
                                                   states_dict[0])

    # print("\nNEW LIST OF STATES PER SONG")
    # pprint(new_list_of_song_states)

    trajectories = get_trajectories(new_list_of_song_states, states_dict,
                                    actions_dict)
    pprint("\nTRAJECTORIES")
    pprint(trajectories)
    save_obj(trajectories, "TRAJECTORIES")

    start_states = get_start_states(trajectories)
    print("\nSTART_STATES")
    pprint(start_states)
    save_obj(start_states, "START STATES")

    terminal_states = get_terminal_states(trajectories)
    print("\nTERMINAL_STATES")
    pprint(terminal_states)
    save_obj(terminal_states, "TERMINAL STATES")


    # states_with_int_elem = convert_elem_of_states_to_int(all_states)
    # states_of_int = convert_states_with_int_elem_to_int(states_with_int_elem,
    #                                                     elem_set_sizes)
    #
    # pprint.pprint(states_of_int)
    # pprint.pprint(states_with_int_elem)
    # pprint.pprint(dict_int_states_to_states)