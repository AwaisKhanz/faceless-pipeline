REFERENCE CLIPS
===============

Narration is cloned from a short recording of a real voice. This folder holds
those recordings, one folder per language.


WHERE THINGS GO
---------------

    voices_refs/
      en/    English clips
      de/    German clips
      es/    Spanish clips
      ...    any of the 23 supported languages, by code

Each language has its own choice, so English can be read by one voice and
German by another. A clip in en/ is only offered when you are choosing an
English voice.

Files left loose in voices_refs/ still work - they appear under "Not sorted
yet" and are offered for every language. The Voices panel has a button that
files them away when the name makes the language obvious.


ADDING A VOICE  (the whole process)
-----------------------------------

1. Get a clip you hold the rights to. See RIGHTS below.

2. Name it for how it sounds, not what it is:

       warm-documentary-male.mp3     shows as "Warm documentary male"
       calm-older-female.mp3         shows as "Calm older female"
       ruhige-erzaehlerin.mp3        shows as "Ruhige erzaehlerin"

   Dashes or underscores both work. The filename is the identity; the label
   is just tidied up for display.

3. Put it in the folder for its language:

       voices_refs/en/warm-documentary-male.mp3

4. Reload the Voices panel. The clip appears under that language.

5. Press Preview. It reads a real line from your own script, not "hello
   world" - how a voice handles your actual writing is the only thing worth
   judging.

6. Adjust Expression and Guidance if you want, then press "Use this".

7. If you changed the voice for a language that already has narration,
   re-voice it. Old audio stays cached under the previous voice and would
   otherwise be reused. Project page -> that language -> Redo.


WHAT MAKES A GOOD CLIP
----------------------

    30+ seconds        clones far better than 10; under 8s is flagged
    one speaker        no interviews, no overlapping voices
    clean              no music, no background noise, no heavy reverb
    the right pace     it copies delivery, so use the speed you want
    plain speech       ordinary sentences, not shouting or whispering

Format does not matter much: .wav .mp3 .m4a .flac .ogg .aac all work. The
pipeline makes its own normalised copy in cache/refs/ - mono, 24 kHz, silence
trimmed, levels evened. You never need to edit anything yourself.


RIGHTS
------

Whatever you put here becomes your channel's voice, on every video, publicly.
It has to be audio you may use that way:

    your own voice          30 seconds on a phone is plenty, and nobody
                            else on YouTube sounds like you

    Mozilla Common Voice    https://commonvoice.mozilla.org/
                            released CC0 - an explicit public-domain
                            dedication. Thousands of speakers, many languages.

    LibriVox                https://librivox.org/
                            public-domain audiobook readings

Audio from a paid AI voice service is not usable - it breaks those services'
terms. Nor is audio lifted from someone else's video: that is a person's
voice and likeness, and they did not agree to narrate your channel.


HOUSEKEEPING
------------

Nothing in this folder is generated. Normalised copies live in cache/refs/
and are rebuilt automatically whenever a source clip changes, so that cache
can be deleted at any time.

Deleting a clip a language is currently using will not break anything
immediately - you get a clear message next time you voice, asking you to
choose another.
