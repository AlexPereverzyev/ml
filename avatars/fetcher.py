import os.path
import config
import logging
from task import TaskResult
from folder_tree_builder import FolderTreeBuilder
from avatar_searcher import AvatarSearcher
from avatar_loader import AvatarLoader
from fb_client import FacebookClient

_script_name = os.path.splitext(os.path.basename(__file__))[0]
_logger = logging.getLogger(_script_name)
_logger.debug('initializing pipeline')

_client = FacebookClient(
    logging.getLogger(FacebookClient.__name__),
    config.current.access_token)

_pipeline = [
    FolderTreeBuilder(
        config.current,
        logging.getLogger(FolderTreeBuilder.__name__)),
    AvatarSearcher(
        config.current,
        logging.getLogger(AvatarSearcher.__name__),
        _client),
    AvatarLoader(
        config.current,
        logging.getLogger(AvatarLoader.__name__),
        _client)
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
