#! python3

import os.path
import logging
from task import TaskResult
from folder_tree_builder import FolderTreeBuilder
from avatar_converter import AvatarConverter
from avatar_searcher import AvatarSearcher
from avatar_loader import AvatarLoader


_script_name = os.path.splitext(os.path.basename(__file__))[0]
_logger = logging.getLogger(_script_name)
_logger.debug('initializing pipeline')
_pipeline = [
    FolderTreeBuilder(),
    AvatarSearcher(),
    AvatarLoader(),
    AvatarConverter()
]
_logger.debug('running pipeline')

for task in _pipeline:
    try:
        result = task.start()
    except Exception:
        _logger.exception('task {0} failed to execute'.format(task))
        _logger.critical('terminating due to previous errors')
        break
    if result in [TaskResult.UNKNOWN, TaskResult.FAIL]:
        _logger.error('task {0} exited with inappropriate code'.format(task))
        _logger.critical('terminating due to previous errors')
        break
else:
    _logger.debug('all done')
