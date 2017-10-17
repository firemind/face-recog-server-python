import os
import json
import cognitive_face as CF
import time

import math
from cognitive_face import CognitiveFaceException


class FaceAPI:
    CALL_LIMIT_PER_MINUTE = 20
    IDENTIFY_FACE_LIMIT = 10

    KEY = '8ee57766bc6040b98a1be5ec498e705f'  # Replace with a valid subscription key (keeping the quotes in place).
    CF.Key.set(KEY)

    BASE_URL = 'https://westeurope.api.cognitive.microsoft.com/face/v1.0/'
    CF.BaseUrl.set(BASE_URL)

    def __init__(self, group_id):
        self.group_id = group_id

    def __cache(self, file_path, action, call):
        cache_file = file_path + "." + action + ".json"
        if os.path.isfile(cache_file):
            with open(cache_file, 'r') as f:
                result = json.load(f)
        else:
            result = call()
            with open(cache_file, 'w') as f:
                json.dump(result, f)
        return result

    def wait(self, return_value):
        time.sleep(4)
        return return_value

    def detect_face(self, image_path):
        return self.__cache(image_path, "detect",
                            lambda: self.wait(CF.face.detect(image_path)))

    def create_person_group(self):
        return self.wait(CF.person_group.create(self.group_id))

    def create_person(self, root, dirname):
        return self.__cache(os.path.join(root, dirname), "create_person",
                            lambda: self.wait(CF.person.create(self.group_id, dirname)))

    def train(self):
        CF.person_group.train(self.group_id)
        while True:
            result = self.wait(CF.person_group.get_status(self.group_id))
            print result
            if result["status"] == "succeeded":
                return

    def reset(self, path):
        try:
            self.wait(CF.person_group.delete(self.group_id))
            print ("deleting %s" % self.group_id)
        except CognitiveFaceException:
            print("could not delete Person Group!")
            pass
        os.system("find " + path + " -name '*.json' | xargs rm")

    def addFace(self, imagePath, personId, target_face):
        return self.__cache(imagePath, "add_face",
                            lambda: self.wait(CF.person.add_face(imagePath, self.group_id, personId, target_face=target_face)))

    def list(self):
        return self.wait(CF.person.lists(self.group_id))

    def identify(self, faceIds):
        results = []
        for i in range(int(math.ceil(float(len(faceIds)) / FaceAPI.IDENTIFY_FACE_LIMIT))):
            results.extend(
                self.wait(
                    CF.face.identify(faceIds[i * FaceAPI.IDENTIFY_FACE_LIMIT:(i + 1) * FaceAPI.IDENTIFY_FACE_LIMIT],
                                     self.group_id)))
        return results
