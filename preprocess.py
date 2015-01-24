__author__ = 'redhat'

from music21 import converter as c
import os

song = c.parse('corpus/0 Christus,_Der_Ist_Mein_Leben.xml')

states = []
first = True
for stream in song.parts[0]:
    if first:
        # skip the Instrument stream.
        first = False
        continue

    if stream.number == 1 and stream.duration != stream.barDuration:
        # it's an anacrusis
        print('anacrusis')
        bar = stream.notesAndRests
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


    bar = stream.notesAndRests

    for elem in bar:
        if elem.isChord:
            # assume elem is Chord
            try:
                prev_fig_chord = chord
            except NameError:
                pass
            chord = elem.figure
            is_fighead = 1

        elif elem.isNote:
            # elem is a note

            # when at new fighead
            if is_fighead == 1 or elem.beat == 1.0:
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

for state in states:
    print(state)
print('fig_notes, chord, fig_duration, fig_start_at_beat, current_fighead')