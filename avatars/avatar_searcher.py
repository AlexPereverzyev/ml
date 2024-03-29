
import json
from task import Task, TaskResult
from fb_client import FacebookClient
from di import inject


class AvatarSearcher(Task):
    @inject
    def __init__(self, client: FacebookClient):
        super().__init__()
        self.client = client

    def routine(self):
        for term in self.config.search_terms:
            size = self.config.search_results.page_size
            start = self.config.search_results.offset // size
            offset = self.config.search_results.offset
            for p in range(start + 1,
                           start + self.config.search_results.page_count + 1):
                term_dir, _, _, ids_map_file = self.paths_for(term, p)
                if self.exists(ids_map_file):
                    continue
                try:
                    self.logger.info(
                        'searching for users: \'{0}\', page {1}'
                        .format(term, p))
                    ids = self.client.search(term, offset, size)
                except Exception:
                    self.logger.exception(
                        'user search failed: \'{0}\', page {1}'
                        .format(term, p))
                else:
                    if not ids:
                        self.logger.info('nothing found for: \'{0}\', page {1}'
                                         .format(term, p))
                        break
                    ids_map = {id: self.default_map for id in ids}
                    ids_map = json.dumps(ids_map, indent=4)
                    with open(ids_map_file, 'w+') as f:
                        f.write(ids_map)
                    self.logger.info(
                        'saved ids map to: {0}'.format(ids_map_file))
                offset += size
        return TaskResult.OK
