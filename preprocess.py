__author__ = 'redhat'

from music21 import converter, note, harmony, stream
from common_methods import *
import itertools
import shutil

# NEED TO REDUCE K

def parse(filename):
    """
    Idea:
    1. When it's a chord, the next note is a figurehead
    2. Continue acquiring all notes that falls into the same figure, until
    3. meeting the next chord, which implies new figure
    4. Pickup bar and the last figure are handled separately

    notes:
    1. In 4/4 beat occurs every quarter.
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

        states.append((elem.measureNumber, tuple(pickup), '0', pickup_beat,
                       pickup_duration, pickup_fighead))

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
            if prev_elem.isChord or prev_elem.isRest or elem.beat == 1.0:
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
                figure = (elem.measureNumber,  #  0
                          tuple(fig_notes),    #  1
                          fig_chord,           #  2
                          fig_start_at_beat,   #  3
                          fig_duration,        #  4
                          fighead)             #  5
                states.append(figure)

        elif elem.isRest and not last_item:
            # elem is a rest
            states.append((elem.measureNumber,
                           ('rest', elem.quarterLength),
                           'rest',
                           elem.beat,
                           elem.quarterLength,
                           -1))

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

    try:
        elems = sorted(list_of_elem)
    except TypeError:
        elems = list_of_elem


    # assume that all tuples have same length
    dict_elem_to_int = {}  # key: elem, value: int
    dict_int_to_elem = {}  # key: int, value: elem
    counter = 1
    for elem in elems:
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
            new_state = [state[0],
                         fignotes_dict[0][state[1]],
                         chords_dict[0][state[2]],
                         int(state[3] * 4),
                         int(state[4] * 4),
                         state[5]]
            new_states.append(tuple(new_state))
        new_all_states.append(new_states)
    return new_all_states


def compute_action(s_prime):
    # find a that makes fs transition to s_prime
    # no bar in action
    action = s_prime[1:3] + s_prime[-2:]
    return action


def get_trajectories(list_of_song_states):
    trajectories = []
    for states in list_of_song_states:
        trajectory = []
        first = True
        for state in states:
            if first:
                trajectory.append(state)
                first = False
                continue
            action = compute_action(state)
            trajectory.append(action)
            trajectory.append(state)
        trajectory.append(-1)  # exit action for terminal state
        trajectories.append(trajectory)
    save_obj(trajectories, 'TRAJECTORIES')
    return trajectories

def get_terminal_states(trajectories):
    terminal_states = []
    for trajectory in trajectories:
        terminal_states.append(trajectory[-2])
    save_obj(terminal_states, 'TERM_STATES')
    return set(terminal_states)


def get_start_states(trajectories):
    start_states = []
    for trajectory in trajectories:
        start_states.append(trajectory[0])
    save_obj(start_states, 'START_STATES')
    return set(start_states)


def get_all_states(new_list_of_song_states):
    # get all combinations of possible stae
    all_states = make_flat_list(new_list_of_song_states)
    # do something
    save_obj(all_states, 'ALL_STATES')
    return all_states

def get_all_actions(all_states):
    all_actions = []
    for state in all_states:
        if state[2] == 1:
            # pickup
            continue
        else:
            all_actions.append(state[1:3] + state[-2:])
    save_obj(all_actions, 'ALL_ACTIONS')
    return all_actions

def preprocess():
    if os.path.exists('obj'):
        shutil.rmtree('obj')

    filenames = get_corpus('corpus/')
    list_of_song_states = []
    for filename in filenames:
        states = parse(filename)
        list_of_song_states.append(states)

    fignotes = []
    chords = []
    figheads = []
    for states in list_of_song_states:
        for state in states:
            fignotes.append(state[1])
            chords.append(state[2])
            figheads.append(state[-1])


    fignotes_dict = make_dict_of_elem(fignotes, 'FIGNOTES_DICT')
    chords_dict = make_dict_of_elem(chords, 'CHORDS_DICT')

    new_list_of_song_states = convert_each_elem_to_int(list_of_song_states,
                                                       fignotes_dict,
                                                       chords_dict)
    trajectories = get_trajectories(new_list_of_song_states)
    start_states = get_start_states(trajectories)
    terminal_states = get_terminal_states(trajectories)

    all_states = get_all_states(new_list_of_song_states)
    all_actions = get_all_actions(all_states)


    # print('\nFIGNOTES_DICT')
    # pprint(fignotes_dict)
    # print('\nCHORDS_DICT')
    # pprint(chords_dict)
    # print('\nALL_STATES')
    # pprint(all_states)
    # print('\nALL_ACTIONS')
    # pprint(all_actions)
    # print('\nTRAJECTORIES')
    # pprint(trajectories)
    # print('\nSTART_STATES')
    # pprint(start_states)
    # print('\nTERMINAL_STATES')
    # pprint(terminal_states)

if __name__ == "__main__":
    preprocess()