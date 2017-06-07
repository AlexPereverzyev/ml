
import os
from PIL import Image
from task import Task, TaskResult


class AvatarConverter(Task):

    def routine(self):
        for term in self.config.search_terms:
            _, pic_dir, pic_proc_dir, _ = self.paths_for(term)
            for avatar in (p for p in os.listdir(pic_dir)
                           if p.endswith(self.config.picture_format)):
                avatar_file = os.path.join(pic_dir, avatar)
                proc_avatar_file = os.path.join(pic_proc_dir, avatar)
                if self.exists(proc_avatar_file):
                    continue
                self.logger.info('processing avatar: {0}'.format(avatar_file))
                pic = Image.open(avatar_file)
                pic = pic.resize(self.config.picture_size)
                pic = pic.convert('L')
                pic.save(proc_avatar_file)
                self.logger.info('saved avatar to: {0}'
                                 .format(proc_avatar_file))
        return TaskResult.OK
