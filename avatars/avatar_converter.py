
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
                    pic_proc_dir, os.path.splitext(avatar)[0])
                if self.exists(proc_avatar_file):
                    continue
                self.logger.info('processing avatar: {0}'.format(avatar_file))
                self._process_avatar(avatar_file, proc_avatar_file)
        return TaskResult.OK

    def _process_avatar(self, source, destination):
        img = Image.open(source)
        for i, (c, b) in enumerate(self.detector.find_unique(img)):
            match = img.crop(b)
            match_name = '{0}-{1}.{2}'.format(
                destination, i + 1, self.config.picture_format)
            match.save(match_name)
            self.logger.info('found match: {0:.2f}, {1}'.format(c, b))
            self.logger.info('face saved to: {0}'.format(match_name))

    def _is_image(self, filename):
        return any(filename.lower().endswith('.' + f)
                   for f in self.known_formats)
