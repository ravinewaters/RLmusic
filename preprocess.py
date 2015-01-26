__author__ = 'redhat'

from music21 import converter as c
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

    song = c.parse(filename)

    states = []
    measures = song.parts[0].getElementsByClass("Measure")
    number_of_measures = len(measures)
    for measure in measures:

        # Anacrusis
        if measure.number == 1 and measure.duration != measure.barDuration:
            bar = measure.notesAndRests
            pickups = []

            first = True
            for elem in bar:
                if elem.isNote:
                    if first:
                        fighead = elem.midi
                        first = False
                    pickups.append(elem.midi)
                    pickups.append(elem.quarterLength)
                elif elem.isRest:
                    pickups.append('rest')
                    pickups.append(elem.quarterLength)
            states.append(("pickup", tuple(pickups)))
            continue

        for elem in measure.notesAndRests:

            if elem.isChord:
                # elem is a Chord
                try:
                    prev_fig_chord = chord
                except NameError:
                    pass

                # get chord's name
                chord = elem.figure
                is_fighead = 1

            elif elem.isNote:
                # elem is a note
                # when at new fighead or last note, i.e. the next object is
                # the final barline

                # NEED TO HANDLE TIE AS WELL
                if is_fighead == 1 and elem.beat == 1.0:
                    # many variables are undefined when iterating the first bar

                    try:
                        # if fig_start_at_beat undefined, define it for the new
                        # figure
                        prev_fig_start_at_beat = fig_start_at_beat
                        prev_fig_duration = fig_duration
                        prev_fig_notes = fig_notes
                        prev_fighead = current_fighead

                        states.append((tuple(prev_fig_notes),
                                       prev_fig_chord,
                                       prev_fig_duration,
                                       prev_fig_start_at_beat,
                                       prev_fighead))
                    except NameError:
                        pass

                    current_fighead = elem.midi  # set new fighead
                    fig_duration = 0
                    fig_start_at_beat = elem.beat  # determine beat of the new
                    # figure
                    fig_notes = []  #set empty for the new figure
                    is_fighead = 0  # next note may not be a fighead

                #append notes to fig_notes
                fig_notes.append(elem.midi)
                fig_notes.append(elem.quarterLength)
                fig_duration += elem.quarterLength

            elif elem.isRest:
                # elem is a rest
                states.append(('rest', elem.quarterLength, elem.beat))

            # terminal state, last figure
            if measure.number == number_of_measures and not hasattr(elem.next(),
                                                               'pitches'):
                prev_fig_start_at_beat = fig_start_at_beat
                prev_fig_duration = fig_duration
                prev_fig_notes = fig_notes
                prev_fighead = current_fighead

                states.append(("end",
                               tuple(prev_fig_notes),
                               chord,
                               prev_fig_duration,
                               prev_fig_start_at_beat,
                               prev_fighead))

    # for state in states:
    #     print(state)
    # print('fig_notes, chord, fig_duration, fig_start_at_beat, current_fighead')

    return states

def get_corpus(corpus_dir):
    filenames = []
    for f in os.listdir(corpus_dir):
        if '.xml' in f and os.path.isfile(corpus_dir + f):
            filenames.append(corpus_dir + f)
    return filenames

if __name__ == "__main__":
    filenames = get_corpus('corpus/')
    for filename in filenames:
        print('\n' + filename)
        pprint.pprint(parse(filename))