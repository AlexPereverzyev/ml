
import os
from task import Task, TaskResult


class FolderTreeBuilder(Task):

    def routine(self):
        self._create_dir_safe(self.config.storage_path)
        for term in self.config.search_terms:
            term_dir, pic_dir, pic_proc_dir, _ = self.paths_for(term)
            self._create_dir_safe(term_dir)
            self._create_dir_safe(pic_dir)
            self._create_dir_safe(pic_proc_dir)
        return TaskResult.OK

    def _create_dir_safe(self, dirname):
        if not os.path.exists(dirname):            
            os.makedirs(dirname)
            self.logger.info('created directory: {0}'.format(dirname))
