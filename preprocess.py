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

            # NEED TO HANDLE TIE AS WELL

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

def generate_features(states):
    pass

if __name__ == "__main__":
    filenames = get_corpus('corpus/')
    for filename in filenames:
        print('\n' + filename)
        pprint.pprint(parse(filename))