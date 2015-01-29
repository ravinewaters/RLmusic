__author__ = 'redhat'

from music21 import converter, note, harmony, stream
import os
import pprint

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


    for i in range(len(elements)):
        prev_elem = elements[i-1]
        elem = elements[i]
        try:
            next_elem = elements[i+1]
        except stream.StreamException:
            # at the last iteration, look for next item in the original measure
            next_elem = elem.next()

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

        elif elem.isRest:
            # elem is a rest
            states.append((('rest',), 'rest', elem.quarterLength, elem.beat,
                           'rest'))

    return states

def get_corpus(corpus_dir):
    filenames = []
    for f in os.listdir(corpus_dir):
        if '.xml' in f and os.path.isfile(corpus_dir + f):
            filenames.append(corpus_dir + f)
    return filenames

def make_dict_of_elem(list_of_tuples):
    """
    Assume that we have a list of tuples.
    An element in n-th coordinate of a tuple comes from a set.
    The set contains all elements that are in n-th coordinate of all
    tuples in the list.
    Map all the elements from the set to a nonnegative integer and vice versa.
    Returns 3 lists.
    The first list contains dictionaries of k-th element to nonnegative integer
    The second is the other way around.
    the third contains size of each set
    """

    # initialize list of dictionaries
    list_to_int = []
    list_to_elem = []
    list_size = []

    # assume that all tuples have same length
    for i in range(len(list_of_tuples[0])):
        list_to_int.append({})  # key: elem, value: int
        list_to_elem.append({})  # key: int, value: elem
        counter = 0
        for tuple in list_of_tuples:
            # unpack elements in the tuple
            # store it in dictionary if the key doesn't exist, pass otherwise
            if tuple[i] not in list_to_int[i]:
                list_to_int[i][tuple[i]] = counter
                list_to_elem[i][counter] = tuple[i]
                counter += 1
        list_size.append(counter)
    return (list_to_int, list_to_elem, list_size)

def convert_elem_of_states_to_int(list_of_lists_of_tuples):
    """
    returns states that are tuple of nonnegative integers, e.g.
    (1,1,1,1,2) instead of ((0, 1.0), 'C', 1.0, 1.0, 0)
    """

    # flatten list of list of tuples to list of tuples
    flatten_states = [item for list_of_tuples in
                      list_of_lists_of_tuples for item in list_of_tuples]

    global dict_elem_to_int, dict_int_to_elem, elem_set_sizes
    dict_elem_to_int, dict_int_to_elem, elem_set_sizes = \
        make_dict_of_elem(flatten_states)

    new_all_states = []
    for states in list_of_lists_of_tuples:
        new_states = []
        for state in states:
            new_state = []
            for i in range(len(state)):
                new_elem = dict_elem_to_int[i][state[i]]
                new_state.append(new_elem)
            new_states.append(tuple(new_state))
        new_all_states.append(new_states)
    return new_all_states

def convert_states_with_int_elem_to_int(list_of_lists_of_tuples, elem_set_sizes):
    pass


def make_action_set(trajectories):
    all_states = [state for trajectory in
                      trajectories for state in trajectory]
    action_set_by_duration = {}
    for state in all_states:
        if state[1] != 'pickup':
            if state[2] in action_set_by_duration:
                    action_set_by_duration[state[2]].append(state[:2])
            else:
                action_set_by_duration[state[2]] = []
    for key in action_set_by_duration:
        action_set_by_duration[key] = set(action_set_by_duration[key])
    return action_set_by_duration

def get_terminal_states(trajectories):
    terminal_states = []
    for trajectory in trajectories:
        if trajectory[-1][1] != 'rest':
            terminal_states.append(trajectory[-1])
    return set(terminal_states)

def get_start_states(trajectories):
    start_states = []
    for trajectory in trajectories:
        if trajectory[0][1] != 'rest':
            start_states.append(trajectory[0])
    return set(start_states)


def generate_features(state, action):
    # need numpy array
    pass

def compute_next_state(int_s, int_a):
    s = states_dict[int_s]
    a = actions_dict[int_a]
    s_prime = (a[0], a[1], sum(a[0][::2]), s[2]+s[3], a[0][0])
    return s_prime


if __name__ == "__main__":
    filenames = get_corpus('corpus/')
    all_states = []
    for filename in filenames:
        states = parse(filename)
        all_states.append(states)
    pprint.pprint(all_states)
    pprint.pprint(make_action_set(all_states))
    pprint.pprint(get_start_states(all_states))
    # states_with_int_elem = convert_elem_of_states_to_int(all_states)
    # states_of_int = convert_states_with_int_elem_to_int(states_with_int_elem,
    #                                                     elem_set_sizes)
    #
    # pprint.pprint(states_of_int)
    # pprint.pprint(states_with_int_elem)
    # pprint.pprint(dict_int_states_to_states)