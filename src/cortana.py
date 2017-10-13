import os
import json
import cognitive_face as CF
import time

import math
from cognitive_face import CognitiveFaceException

FACE_PATH = '/home/viruch/Documents/projects/SA/PersonalFaceData/'
trainFaces = 7
testFaces = 3


class FaceAPI:
    CALL_LIMIT_PER_MINUTE = 20
    IDENTIFY_FACE_LIMIT = 10

    KEY = '8ee57766bc6040b98a1be5ec498e705f'  # Replace with a valid subscription key (keeping the quotes in place).
    CF.Key.set(KEY)

    BASE_URL = 'https://westeurope.api.cognitive.microsoft.com/face/v1.0/'
    CF.BaseUrl.set(BASE_URL)

    def __init__(self, group_id):
        self.group_id = group_id

    def wait(self, return_value):
        time.sleep(4)
        return return_value

    def detect_face(self, image_path):
        return self.wait(CF.face.detect(image_path))

    def create_person_group(self):
        return self.wait(CF.person_group.create(self.group_id))

    def create_person(self, dirname):
        return self.wait(CF.person.create(self.group_id, dirname))

    def train(self):
        CF.person_group.train(self.group_id)
        while True:
            result = self.wait(CF.person_group.get_status(self.group_id))
            print result
            if result["status"] == "succeeded":
                return

    def reset(self):
        try:
            self.wait(CF.person_group.delete(self.group_id))
        except:
            print("could not delete Person Group!")
            pass

    def addFace(self, imagePath, personId, target_face):
        return self.wait(CF.person.add_face(imagePath, self.group_id, personId, target_face=target_face))

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


faceAPI = FaceAPI('my_group')


def getFiles(dirname):
    files = [FACE_PATH + dirname + "/" + f for f in os.listdir(FACE_PATH + dirname) if not f.endswith(".json")]
    files.sort()
    return files


def detectFace(absoluteName):
    result = faceAPI.detect_face(absoluteName)
    rectangles = [x["faceRectangle"] for x in result]
    faces = sorted(rectangles, key=lambda x: x["width"] * x["height"])
    if len(faces) == 0:
        print("not found for " + absoluteName)
        return None
    face = faces[0]
    return ",".join(map(str, [face["left"], face["top"], face["width"], face["height"]]))


def upload():
    try:
        faceAPI.create_person_group()
    except CognitiveFaceException:
        pass

    for dirname in os.listdir(FACE_PATH):
        if dirname.endswith(".json"):
            continue
        faces = getFiles(dirname)
        if len(faces) < trainFaces + testFaces:
            print dirname + " has not enough samples: " + str(len(faces))
            continue
        if not os.path.isfile(FACE_PATH + dirname + ".json"):
            result = faceAPI.create_person(dirname)
            with open(FACE_PATH + dirname + '.json', 'w') as outfile:
                json.dump(result, outfile)
        else:
            with open(FACE_PATH + dirname + '.json', 'r') as outfile:
                result = json.load(outfile)
        print(result)
        personId = result["personId"]
        trains = faces[:trainFaces]
        for absoluteName in trains:
            if os.path.isfile(absoluteName + ".json"):
                continue
            target_face = detectFace(absoluteName)
            if target_face is None:
                continue
            result = faceAPI.addFace(absoluteName, personId, target_face)
            print(result)
            with open(absoluteName + '.json', 'w') as outfile:
                json.dump(result, outfile)
            time.sleep(4)


def showServerContent():
    people = faceAPI.list()
    for p in people:
        print p["name"] + ": " + str(len(p["persistedFaceIds"])) + ":" + p["personId"]


def verify():
    testFaceIds = []
    correctFaceToPersonIds = {}

    for dirname in os.listdir(FACE_PATH):
        if dirname.endswith(".json"):
            continue
        faces = getFiles(dirname)
        if len(faces) < trainFaces + testFaces:
            print dirname + " has not enough samples: " + str(len(faces))
            continue
        tests = faces[trainFaces: trainFaces + testFaces]
        for absoluteName in tests:
            if not os.path.isfile(absoluteName + ".json"):
                result = faceAPI.detect_face(absoluteName)
                with open(absoluteName + '.json', 'w') as outfile:
                    json.dump(result, outfile)
                time.sleep(4)
            else:
                with open(absoluteName + '.json', 'r') as outfile:
                    result = json.load(outfile)
            print (result)
            if len(result) == 0:
                print "no face found: " + absoluteName
                continue
            currentFace = {}
            currentFace["fileName"] = absoluteName
            currentFace["person"] = dirname
            face_id = result[0]["faceId"]
            correctFaceToPersonIds[face_id] = currentFace
            testFaceIds.append(face_id)
            with open(FACE_PATH + dirname + '.json', 'r') as outfile:
                currentFace["personId"] = json.load(outfile)["personId"]

    print correctFaceToPersonIds

    results = faceAPI.identify(testFaceIds)

    for r in results:
        face_id = r["faceId"]
        predicted = r["candidates"]
        if len(predicted) == 0:
            print("no prediction: " + correctFaceToPersonIds[face_id])


faceAPI.reset()
upload()
faceAPI.train()
# verify()

showServerContent()
