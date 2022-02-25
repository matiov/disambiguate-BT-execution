"""Test the interaction with the microphone."""

# Copyright (c) 2022, ABB
# All rights reserved.
#
# Redistribution and use in source and binary forms, with
# or without modification, are permitted provided that
# the following conditions are met:
#
#   * Redistributions of source code must retain the
#     above copyright notice, this list of conditions
#     and the following disclaimer.
#   * Redistributions in binary form must reproduce the
#     above copyright notice, this list of conditions
#     and the following disclaimer in the documentation
#     and/or other materials provided with the
#     distribution.
#   * Neither the name of ABB nor the names of its
#     contributors may be used to endorse or promote
#     products derived from this software without
#     specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import pyttsx3
import speech_recognition as sr


def speak(speaker: pyttsx3, sentence: str) -> None:
    """Handle the speaker interaction."""
    print(sentence)
    speaker.say(sentence)
    speaker.runAndWait()
    speaker.stop()


def hello_there(recognizer: sr.Recognizer, speaker: pyttsx3) -> None:
    """Verbal interaction."""
    print('Say: "Hello there!" when you see the message "Listening..." .')
    with sr.Microphone() as source:
        print('Listening...')
        audio = recognizer.listen(source, timeout=20)
        object_sound = recognizer.recognize_google(
            audio, key=None, language='en-US', show_all=False)

        speak(speaker, 'General Kenobi!')

        assert object_sound == 'hello there'


if __name__ == '__main__':

    microphone_name =\
        'alsa_input.usb-0b0e_Jabra_SPEAK_510_USB_1C48F9F35674022000-00.mono-fallback'

    # Speech Recogniotion!
    r = sr.Recognizer()
    r.energy_threshold = 420
    r.dynamic_energy_threshold = False
    # Sound output
    speaker = pyttsx3.init()
    rate = speaker.getProperty('rate')
    speaker.setProperty('rate', rate - 80)

    # ask the user the object to disambiguate!
    hello_there(recognizer=r, speaker=speaker)
