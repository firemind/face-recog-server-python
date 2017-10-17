import os
import json
from face_api import FaceAPI
from cognitive_face import CognitiveFaceException

FACE_PATH = '/home/viruch/Documents/projects/SA/PersonalFaceDataCortana/'
max_test_sample_size = 7
verify_sample_size = 3

faceAPI = FaceAPI('my_group')

def getFiles(dirname):
    return sorted([FACE_PATH + dirname + "/" + f
                   for f in os.listdir(os.path.join(FACE_PATH, dirname))
                   if not f.endswith(".json")])


def extract_rectangle_from_face(absoluteName):
    result = faceAPI.detect_face(absoluteName)
    rectangles = [x["faceRectangle"] for x in result]
    faces = sorted(rectangles, key=lambda x: x["width"] * x["height"])
    if len(faces) == 0:
        print("not found for " + absoluteName)
        return None
    face = faces[0]
    return ",".join(map(str, [face["left"], face["top"], face["width"], face["height"]]))


def split_data_set(test_sample_size, verify_sample_size):
    test_set = []
    verify_set = []
    for dirname in sorted([p for p in os.listdir(FACE_PATH)
                           if not p.endswith(".json")]):
        files = getFiles(dirname)
        if len(files) < test_sample_size + verify_sample_size:
            print dirname + " has not enough samples: " + str(len(files))
            continue
        verify_set.append((dirname, files[:verify_sample_size]))
        test_set.append((dirname, files[verify_sample_size:verify_sample_size+test_sample_size]))
    return test_set, verify_set


def upload(test_set):
    for dirname, test_files in test_set:
        person = faceAPI.create_person(FACE_PATH, dirname)
        personId = person["personId"]
        print("person %s: %s" % (dirname, personId))
        for absoluteName in test_files:
            target_face = extract_rectangle_from_face(absoluteName)
            if target_face is None:
                continue
            faceAPI.addFace(absoluteName, personId, target_face)
            print("adding %s in %s to %s: %s" % (os.path.basename(absoluteName), target_face, dirname, personId))


def showServerContent():
    people = faceAPI.list()
    for p in people:
        print p["name"] + ": " + str(len(p["persistedFaceIds"])) + ":" + p["personId"]


def verify(verify_set):
    verify_faces = {}

    for dirname, tests in verify_set:
        for absoluteName in tests:
            result = faceAPI.detect_face(absoluteName)
            if len(result) == 0:
                print "no face found: " + absoluteName
                continue
            verify_faces[result[0]["faceId"]] = {
                "fileName": absoluteName,
                "person": dirname,
                "personId": faceAPI.create_person(FACE_PATH, dirname)["personId"]
            }

    results = faceAPI.identify(verify_faces.keys())

    correct = 0.0

    for r in results:
        face = verify_faces[r["faceId"]]
        predicted = r["candidates"]
        if len(predicted) == 0:
            print("no prediction: " + face["fileName"])
            continue
        if face["personId"] == predicted[0]["personId"]:
            correct += 1
        else:
            print("missmatch for: " + face + " predicted: " + predicted)
    return correct / len(results)


def saveResults(res):
    with open(os.path.join(FACE_PATH, "results.json"), 'w') as f:
        json.dump(res, f)

def main():
    #faceAPI.reset(FACE_PATH)
    try:
        faceAPI.create_person_group()
    except CognitiveFaceException:
        print "failed to create group: " + faceAPI.group_id
        pass
    res = []
    for test_sample_size in range(1, max_test_sample_size + 1):
        test_set, verify_set = split_data_set(test_sample_size, verify_sample_size)
        upload(test_set)
        faceAPI.train()
        successrate = verify(verify_set)
        print ("test sample size: %i verify sample size: %i successrate: %f" %
               (test_sample_size, verify_sample_size, successrate))
        res.append({"test_sample_size": test_sample_size, "successrate": successrate})
    saveResults(res)


if __name__ == '__main__':
    main()
