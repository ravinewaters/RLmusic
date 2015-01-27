__author__ = 'redhat'

from music21 import converter, note, harmony, stream
import os
import pprint

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
    first_measure = song.parts[0][1]
    for i in range(len(elements)):
        prev_elem = elements[i-1]
        elem = elements[i]
        try:
            next_elem = elements[i+1]
        except stream.StreamException:
            # at the last iteration, look for next item in the original measure
            next_elem = elem.next()

        # Anacrusis
        if elem.measureNumber == 1 and first_measure.duration != \
                first_measure.barDuration:
            pickups = []
            fighead = None

            if elem.isNote:
                if not fighead:
                    fighead = elem.midi

                pickups.append(elem.midi)
                pickups.append(elem.quarterLength)

            elif elem.isRest:
                pickups.append('rest')
                pickups.append(elem.quarterLength)

            states.append(("pickup", tuple(pickups), fighead))
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
            # 1.0)
            if not hasattr(next_elem, 'pitch') or next_elem.beat == 1.0:
                figure = (tuple(fig_notes),
                          fig_chord,
                          fig_duration,
                          fig_start_at_beat,
                          fighead)

                # If next_elem is the final barline, wrap up the figure
                if hasattr(next_elem, 'style'):
                    figure = ('end',) + figure
                states.append(figure)

        elif elem.isRest:
            # elem is a rest
            states.append(('rest', elem.quarterLength, elem.beat))

    return states

def get_corpus(corpus_dir):
    filenames = []
    for f in os.listdir(corpus_dir):
        if '.xml' in f and os.path.isfile(corpus_dir + f):
            filenames.append(corpus_dir + f)
    return filenames

def hash_elem_tuple(list_of_tuples):
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






def to_tuple_of_ints(states):
    """
    returns states that are tuple of nonnegative integers, e.g.
    (1,1,1,1,2) instead of ((0, 1.0), 'C', 1.0, 1.0, 0)
    """
    fignotes_to_int = tuple_elem_to_int(states, 0)[0]
    chord_to_int = tuple_elem_to_int(states, 1)[0]
    duration_to_int = tuple_elem_to_int(states, 2)[0]
    beat_to_int = tuple_elem_to_int(states, 3)[0]
    fighead_to_int = tuple_elem_to_int(states, 4)[0]

    new_states = []
    for state in states:
        fignotes, figchord, figduration, figbeat, fighead = state
        new_states.append((
            fignotes_to_int[fignotes],
            chord_to_int[figchord],
            duration_to_int[figduration],
            beat_to_int[figbeat],
            fighead_to_int[fighead],
        )
        )








def generate_features(states):
    pass

if __name__ == "__main__":
    filenames = get_corpus('corpus/')
    for filename in filenames:
        print('\n' + filename)
        pprint.pprint(parse(filename))