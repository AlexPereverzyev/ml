
import os
from PIL import Image
from matching import FaceDetector
from task import Task, TaskResult
from di import inject


class AvatarConverter(Task):

    known_formats = ['jpg', 'png', 'bmp', 'jfif']

    @inject
    def __init__(self, detector: FaceDetector):
        super().__init__()
        self.detector = detector

    def routine(self):
        for term in self.config.search_terms:
            _, pic_dir, pic_proc_dir, _ = self.paths_for(term)
            for avatar in (f for f in os.listdir(pic_dir)
                           if self._is_image(f)):
                avatar_file = os.path.join(pic_dir, avatar)
                proc_avatar_file = os.path.join(
                    pic_proc_dir,
                    '{0}.{1}'.format(os.path.splitext(avatar)[0],
                                     self.config.picture_format))
                if self.exists(proc_avatar_file):
                    continue
                self.logger.info('processing avatar: {0}'.format(avatar_file))
                if self._process_avatar(avatar_file, proc_avatar_file):
                    self.logger.info('saved face(s) to: {0}'
                                     .format(proc_avatar_file))
        return TaskResult.OK

    def _process_avatar(self, source, destination):
        img, r = Image.open(source), False
        for c, b in self.detector.find_unique(img):
            match, r = img.crop(b), True
            match.save(destination)
            self.logger.info('found match: {0:.2f}, {1}'.format(c, b))
        return r

    def _is_image(self, filename):
        return any(filename.lower().endswith('.' + f)
                   for f in self.known_formats)
