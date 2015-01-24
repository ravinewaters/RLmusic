__author__ = 'redhat'

from music21 import converter as c

song = c.parse('corpus/0 Diademata.xml')

states = []
first = True

for stream in song.parts[0]:
    if first:
        # skip the Instrument stream.
        first = False
        continue

    bar = stream.notesAndRests

    current_fighead = None
    fig_start_at_beat = None
    fig_notes = None

    for elem in bar:
        if elem.isChord:
            # assume elem is Chord
            chord = elem.figure
            print(chord)
            is_fighead = 1
        elif elem.isNote:
            # elem is a note

            # when at new fighead
            if is_fighead == 1 or elem.beat == 0.0:

                if fig_start_at_beat:
                    # if fig_start_at_beat is already set
                    fig_duration = (elem.beat - fig_start_at_beat) % 4
                else:
                    # if not, current beat is where the figure starts
                    fig_start_at_beat = elem.beat

                # get figure notes and set it empty for next figure
                prev_fig_notes = fig_notes
                fig_notes = []

                # if fighead is already defined
                if current_fighead:
                    states.append((tuple(prev_fig_notes),
                                   chord,
                                   fig_duration,
                                   fig_start_at_beat,
                                   current_fighead))
                current_fighead = elem.midi
                is_fighead = 0
            fig_notes.append(elem.midi)
        elif elem.isRest:
            # elem is a rest
            print('rest')

for state in states:
    print(state)
print('fig_notes, chord, fig_duration, fig_start_at_beat, current_fighead')