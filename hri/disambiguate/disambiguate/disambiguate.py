"""Main script running the disambiguate framework."""

# Copyright (c) 2021 Fethiye Irmak Doğan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os.path
from typing import Tuple, List
import copy

from disambiguate import ashdetr, fbdetr, mdetr, reg, utils
import matplotlib.pyplot as plt
import pyttsx3
import spacy
from spacy.util import raise_error
import speech_recognition as sr


def speak(speaker, sentence):
    """Handle the speaker interaction."""
    print(sentence)
    speaker.say(sentence)
    speaker.runAndWait()
    speaker.stop()


def disambiguate_scene(
    target_image: str,
    target_object: str = '',
    verbal: bool = True,
    output_imgs: bool = False,
) -> Tuple[bool, List[int], str]:
    """
    Disambiguate the target image using verbal-HRI.

    Args
    ----
        target_image: name of the image to disambiguate.
        target_object: the object to disambiguate.
        verbal: whether to use verbal interaction or not.
        output_imgs: wheter to show the images output during the process.

    Returns
    -------
        disambiguated: if the scene is disambiguated or not.
        bounding_box: bounding box [xmin, ymin, XMAX, YMAX].
        obj_call_name: name of the object.

    """
    microphone_name =\
        'alsa_input.usb-0b0e_Jabra_SPEAK_510_USB_1C48F9F35674022000-00.mono-fallback'

    CLASSES = [
            'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
            'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
            'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
            ]

    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    # get the path of the current file
    working_path = os.path.dirname(os.path.realpath(__file__))
    results_directory = os.path.join(working_path, 'data/results')
    source_img = target_image

    # ---- Load Models ----
    # Referring Expression Generation
    referring_exp = reg.REG(working_path, source_img)
    category_index = referring_exp.get_category_index()
    # Add a general category 'object'
    category_index[91] = {'id': 91, 'name': 'object'}

    # Object Detection from Facebook
    facebook = fbdetr.FBDetr(CLASSES, COLORS)
    global_probas, global_bboxes_scaled = facebook.detection(working_path, source_img)

    # Modulated Detection for End-to-End Multi-Modal Understanding
    grad_cam = mdetr.MDetr(working_path, source_img)

    # Object Detection from Ashkamath (Disambiguate?)
    ash = ashdetr.ASHDetr(working_path, source_img, COLORS, output_imgs)

    nlp = spacy.load('en_core_web_sm')

    general_output_dict = utils.construct_detection_dict(
        working_path, source_img, global_probas, global_bboxes_scaled, CLASSES)

    region_names = ['cropped0.jpg', 'cropped1.jpg']
    asked_questions = []
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 0, 0), (255, 255, 255)]

    bounding_box = []

    # Speech Recogniotion!
    r = sr.Recognizer()
    r.energy_threshold = 420
    r.dynamic_energy_threshold = False
    # Sound output
    speaker = pyttsx3.init()
    rate = speaker.getProperty('rate')
    speaker.setProperty('rate', rate - 80)
    # TODO: set this as parameter somewhere
    interactions = 2

    ambiguous = True
    while ambiguous:
        gradCAM_AoI = []
        region_check_items = []

        os.system('pacmd set-default-source ' + microphone_name)

        expr = target_object
        if target_object == '' and verbal is True:
            # ask the user the object to disambiguate!
            speak(speaker, 'Describe the object!')
            with sr.Microphone() as source:
                try:
                    understood = False
                    trials = copy.deepcopy(interactions)
                    while not understood and trials > 0:
                        trials = trials - 1
                        print('Listening...')
                        audio = r.listen(source, timeout=20)
                        object_sound = r.recognize_google(
                            audio, key=None, language='en-US', show_all=False)
                        speak(speaker, 'You said: ' + object_sound)
                        # Ask for confirmation
                        speak(speaker, 'Did I understand right?')
                        print('Listening...')
                        audio = r.listen(source, timeout=10)
                        answer = r.recognize_google(
                            audio, key=None, language='en-US', show_all=False)
                        speak(speaker, 'You said: ' + answer)
                        if answer == 'yes' or answer == 'correct' or answer == 'right':
                            expr = object_sound
                            understood = True
                        else:
                            speak(speaker, 'Please repeat again.')
                    if not understood:
                        raise_error()
                except Exception:
                    print('Fail recognising the speech. Type the object in the terminal.')
                    expr = input('Object: ')
        elif target_object == '' and verbal is False:
            expr = input('Describe the object: ')

        # Just tell the human what the robot is doing.
        if verbal:
            speak(speaker, 'The target object ' + expr + ' is not univocal.')
        else:
            print('The target object ' + expr + ' is not univocal.')

        # Use gradCAM to compute bounding boxes and region of interest from the expression.
        gradCAM_boxes, gradCAM_AoI = grad_cam.detection(expr, gradCAM_AoI)
        # Use Natural Language Processing to elaborate the expression.
        doc = nlp(expr)
        nouns = [chunk.text for chunk in doc.noun_chunks]
        if not nouns:
            nouns = ['object']
        items_in_expresison = utils.find_items_in_expression(nouns, CLASSES)

        countval = 0
        for j, region in enumerate(gradCAM_AoI):
            for i, box in enumerate(global_bboxes_scaled):
                x0_i = int(box[0])
                y0_i = int(box[1])
                x1_i = int(box[2])
                y1_i = int(box[3])
                if utils.intersects(region, [x0_i, x1_i, y0_i, y1_i]) > 0 and\
                   CLASSES[global_probas[i].argmax()] in items_in_expresison:
                    region_check_items.append((region_names[j], countval))
                    break
            countval += 1

        # if there is only one item with same class as target, identify it and return
        if len(region_check_items) == 1:
            bounding_box = ash.target_bounding_box(
                os.path.join(results_directory, region_check_items[0][0]),
                expr,
                nouns,
                region_check_items[0][1],
                items_in_expresison,
                copy.deepcopy(general_output_dict),
                category_index,
                gradCAM_AoI
            )
            ambiguous = False
            obj_call_name = expr
        # otherwise need to disambiguate
        else:
            if output_imgs:
                show_detected_objs = plt.imread(os.path.join(results_directory, source_img))
                plt.imshow(show_detected_objs)
                plt.show()

            output_dict = copy.deepcopy(general_output_dict)
            target_item = utils.get_target_item_name(
                nouns, items_in_expresison, output_dict, category_index)
            candidate_centers, appended, obj_call_name = utils.handle_hri(
                region_check_items,
                working_path,
                source_img,
                output_dict,
                target_item,
                category_index,
                gradCAM_boxes,
                colors,
                output_imgs
            )

            solved_with_interaction = False
            target_obj_name = ''
            if nouns and len(nouns[0]) > 4 and nouns[0].lower()[:4] == 'the ':
                target_obj_name = nouns[0].lower()[4:]
            elif nouns:
                target_obj_name = nouns[0].lower()

            for i, center in enumerate(candidate_centers):
                referring_expression = referring_exp.generate_referring_expression(
                    target_obj_name,
                    center[0],
                    center[1],
                    obj_call_name,
                    output_dict
                )
                question = 'Is the ' + target_obj_name + referring_expression + '?'
                if referring_expression in asked_questions:
                    continue
                else:
                    # Ask disambiguation question
                    if verbal:
                        speak(speaker, question)
                        asked_questions.append(referring_expression)

                        os.system('pacmd set-default-source ' + microphone_name)
                        with sr.Microphone() as source:
                            try:
                                understood = False
                                trials = copy.deepcopy(interactions)
                                while not understood and trials > 0:
                                    trials = trials - 1
                                    print('Listening...')
                                    audio = r.listen(source, timeout=10)
                                    answer = r.recognize_google(
                                        audio, key=None, language='en-US', show_all=False)
                                    speak(speaker, 'You said: ' + answer)
                                    if answer == 'yes' or answer == 'correct' or\
                                       answer == 'right' or answer == 'no' or answer == 'wrong':
                                        understood = True
                                    else:
                                        speak(speaker, 'Please repeat again.')
                                if not understood:
                                    raise_error()
                            except Exception:
                                print('Fail recognising the speech. Type your answer.')
                                answer = input('Yes, No? ')
                    else:
                        print(question)
                        answer = input('Yes, No? ')

                    if answer.lower() == 'yes' or answer.lower() == 'correct' or\
                       answer.lower() == 'right':
                        solved_with_interaction = True
                        break

            if solved_with_interaction:
                identified_obj = results_directory + '/cropped' + str(i) + '.jpg'
                if appended:
                    bounding_box = ash.target_bounding_box(
                        identified_obj,
                        expr,
                        nouns,
                        i,
                        items_in_expresison,
                        copy.deepcopy(output_dict),
                        category_index,
                        gradCAM_AoI,
                        appended
                    )
                else:
                    bounding_box = ash.target_bounding_box(
                        identified_obj,
                        expr,
                        nouns,
                        i,
                        items_in_expresison,
                        copy.deepcopy(output_dict),
                        category_index,
                        gradCAM_AoI
                    )

                ambiguous = False

    disambiguated = not ambiguous

    return disambiguated, bounding_box, obj_call_name
