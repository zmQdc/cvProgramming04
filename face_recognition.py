import os
import cv2
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import letterbox_image


class FaceRecognition(object):
    _defaults = {
        "input_size": [200, 200],
        "classifier": "haarcascade_frontalface_alt2.xml",
        "classifier_base": r"D:\software\Anaconda3\Lib\site-packages\cv2\data",
        "n_components": 150
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.input_size = self._defaults["input_size"]
        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(self._defaults["classifier_base"], self._defaults["classifier"])
        )
        self.pca = PCA(n_components=self._defaults["n_components"])
        self.model = cv2.face.FisherFaceRecognizer_create()
        self.images = []
        self.labels = []

    def detect_faces(self, img):
        if img.ndim == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img.copy()
        faces = self.face_cascade.detectMultiScale(img_gray)
        results = []
        for (x, y, width, height) in faces:
            results.append([x, y, x + width, y + height])
        return results

    def save_rectangle(self, base: str, train_txt: str):
        if os.path.exists(train_txt):
            os.remove(train_txt)
        fp = open(train_txt)
        cnt = 0
        res = ''
        with open(os.path.join(base, 'val_label.txt'), 'r') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            img_path, label = line.strip().split(' ')
            img = cv2.imread(os.path.join(base, 'img/' + img_path))
            results = self.detect_faces(img)
            cnt += 1
            res += img_path + ' ' + label + ' '
            if len(results) > 0:
                for tmp in results:
                    res += ','.join(str(x) for x in tmp) + ' '
            res += '\n'
        fp.write(res)
        fp.close()
        print('Faces were detected: %.4f' % (cnt / len(lines)))

    def train(self, train_txt: str, base: str):
        if not os.path.exists(train_txt):
            self.save_rectangle(base, train_txt)
        image_data = []
        with open(train_txt, 'r') as f:
            for line in tqdm(f.readlines()):
                tmp = line.strip().split(' ')
                img = cv2.imread(os.path.join(base, 'img/' + tmp[0]))
                label = tmp[1]
                if len(tmp) > 2:
                    rec = list(map(int, tmp[2].split(',')))
                    img = img[rec[1]:rec[3], rec[0]:rec[2]]
                self.images.append(letterbox_image(img, self.input_size))
                self.labels.append(int(label))
        for image in self.images:
            data = image.flatten()
            image_data.append(data)
        x = np.array(image_data)
        y = np.array(self.labels)
        if len(x) != len(y):
            return
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        self.pca.fit(x)
        x_train = self.pca.transform(x_train)
        x_test = self.pca.transform(x_test)
        self.model.train(x_train, y_train)
        hit = 0
        for i in range(len(y_test)):
            if self.model.predict(x_test[i])[0] == y_test[i]:
                hit += 1
        print('train test acc: %.4f' % (hit / len(y_test)))
        for i in range(len(y_train)):
            if self.model.predict(x_train[i])[0] == y_train[i]:
                hit += 1
        print('train acc: %.4f' % (hit / (len(y_test) + len(y_train))))

    def predict(self, base: str, res_path: str):
        files = os.listdir(base)
        image_data = []
        img_names = []
        cnt = 0
        for file in tqdm(files):
            if os.path.isfile(os.path.join(base, file)) and file.endswith('.jpg'):
                img = cv2.imread(os.path.join(base, file))
                results = self.detect_faces(img)
                if len(results) > 0:
                    x1, y1, x2, y2 = results[0]
                    img = img[y1:y2, x1:x2]
                img = letterbox_image(img, self.input_size)
                image_data.append(img.flatten())
                img_names.append(file)
                cnt += 1
        x = np.array(image_data)
        x = self.pca.transform(x)
        res = ''
        for i in range(len(img_names)):
            res += img_names[i] + ' '
            res += str(self.model.predict(x[i])[0]) + '\n'
        fp = open(res_path, 'w')
        fp.write(res)
        fp.close()
        print('cnt: %d, imgs: %d' % (cnt, len(files)))


if __name__ == '__main__':
    face_recognition = FaceRecognition()
    face_recognition.train(train_txt='data/val.txt', base='data')
    face_recognition.predict('data/img', 'data/test_predict.txt')
