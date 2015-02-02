__author__ = 'redhat'

from music21 import converter, note, harmony, stream
from common_methods import *
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

def make_dict_of_elem(list_of_elem, filename):
    # save to filename
    # initialize list of dictionaries
    output = []

    # assume that all tuples have same length
    dict_elem_to_int = {}  # key: elem, value: int
    dict_int_to_elem = {}  # key: int, value: elem
    counter = 2
    for elem in list_of_elem:
        # store it in dictionary if the key doesn't exist, pass otherwise
        if elem not in dict_elem_to_int:
            dict_elem_to_int[elem] = counter
            dict_int_to_elem[counter] = elem
            counter += 1

    output = [dict_elem_to_int, dict_int_to_elem, counter]
    save_obj(output, filename)

    return output

def convert_each_elem_to_int(list_of_song_states, fignotes_dict, chords_dict):
    """
    returns states that are tuple of nonnegative integers, e.g.
    (1,1,1,1,2) instead of ((0, 1.0), 'C', 1.0, 1.0, 0)
    """

    new_all_states = []
    for states in list_of_song_states:
        new_states = []
        for state in states:
            new_state = [fignotes_dict[0][state[0]],
                         chords_dict[0][state[1]],
                         int(state[2] * 4),
                         int(state[3] * 4),
                         state[4]]
            if state[4] == 'rest':
                new_state[4] = 128
            new_states.append(tuple(new_state))
        new_all_states.append(new_states)
    return new_all_states

def get_all_actions(new_list_of_song_states, pickup_number):
    flat_states = make_flat_list(new_list_of_song_states)
    all_actions = []
    for state in flat_states:
        if state[4] == 128 or state[1] == pickup_number:
            continue
        else:
            all_actions.append((state[0], state[1]))

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

def preprocess():
    filenames = get_corpus('corpus/')
    list_of_song_states = []
    for filename in filenames:
        states = parse(filename)
        list_of_song_states.append(states)

    fignotes = []
    chords = []
    for states in list_of_song_states:
        for state in states:
            fignotes.append(state[0])
            chords.append(state[1])

    fignotes_dict = make_dict_of_elem(fignotes, 'FIGNOTES_DICT')
    chords_dict = make_dict_of_elem(chords, 'CHORDS_DICT')

    new_list_of_song_states = convert_each_elem_to_int(list_of_song_states,
                                                   fignotes_dict,
                                                   chords_dict)

    all_actions = get_all_actions(new_list_of_song_states,
                                  chords_dict[0]['pickup'])
    print('\nFIGNOTES_DICT')
    pprint(fignotes_dict)
    print('\nCHORDS_DICT')
    pprint(chords_dict)
    print('\nNEW_LIST_OF_SONG_STATES')
    pprint(new_list_of_song_states)
    print('\nALL_ACTIONS')
    pprint(all_actions)

    # all_actions = make_list_of_all_actions(list_of_song_states)
    #
    # actions_dict = map_tuples_to_int(all_actions)
    # save_obj(actions_dict, 'ACTIONS_DICT')
    #
    # actions_by_duration_dict = make_action_by_duration_dict(list_of_song_states,
    #                                                       actions_dict)
    # save_obj(actions_by_duration_dict, 'ACTIONS_BY_DURATION_DICT')
    #
    # states_dict = map_tuples_to_int(make_flat_list(list_of_song_states))
    # save_obj(states_dict, 'STATES_DICT')
    #
    # new_list_of_song_states = map_item_inside_list_of_list(list_of_song_states,
    #                                                states_dict[0])
    #
    # trajectories = get_trajectories(new_list_of_song_states, states_dict,
    #                                 actions_dict)
    # save_obj(trajectories, "TRAJECTORIES")
    #
    # start_states = get_start_states(trajectories)
    # save_obj(start_states, "START_STATES")
    #
    # terminal_states = get_terminal_states(trajectories)
    # save_obj(terminal_states, "TERM_STATES")

if __name__ == "__main__":
    preprocess()