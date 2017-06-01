
import os
import enum


class Task(object):

    default_map = {'s': 0, 'url': None}

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger 

    def __str__(self):
        return type(self).__name__

    def start(self):
        self.logger.info('starting')
        status = self.routine()
        self.logger.info('completed')
        return status

    def routine(self):
        raise NotImplementedError

    def paths_for(self, term, page=1):
        term_dir = os.path.join(self.config.storage_path, term.replace(' ', '_'))
        pic_dir = os.path.join(term_dir, self.config.pictures_dir)
        pic_proc_dir = os.path.join(term_dir, self.config.processed_pic_dir)
        ids_map_file = os.path.join(term_dir, self.config.out_file_templ.format(page))
        return term_dir, pic_dir, pic_proc_dir, ids_map_file

    def path_for_avatar(self, term, id, ext):
        term_dir = os.path.join(self.config.storage_path, term.replace(' ', '_'))
        pic_dir = os.path.join(term_dir, self.config.pictures_dir)
        avatar_path = os.path.join(pic_dir, '{0}.{1}'.format(id, ext))
        return avatar_path


class TaskResult(enum.Enum):
    UNKNOWN = 0
    OK = 1
    FAIL = 2
    PARTIAL_OK = 4
