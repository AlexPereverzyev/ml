
import json
from task import Task, TaskResult
from fb_client import FacebookClient
from di import inject


class AvatarLoader(Task):

    ext = 'jpg'

    @inject
    def __init__(self, client: FacebookClient):
        super().__init__()
        self.client = client

    def routine(self):
        for term in self.config.search_terms:
            for p in range(1, self.config.search_results.page_count + 1):
                _, _, _, ids_map_file = self.paths_for(term, p)
                with open(ids_map_file, 'r') as f:
                    ids_map = json.loads(f.read())
                filtered_map = {id: stat for id, stat in ids_map.items()
                                if not stat['s']}
                if not filtered_map:
                    self.logger.info('ids already processed in: {0}'
                                     .format(ids_map_file))
                    continue
                for id, stat in filtered_map.items():
                    try:
                        self.logger.info('fetching avatar for: {0}'.format(id))
                        avatar, url = self.client.avatar(id)
                        stat['s'], stat['url'] = 1, url
                    except Exception:
                        self.logger.exception('failed to fetch avatar for: {0}'
                                              .format(id))
                    else:
                        avatar_file = self.path_for_avatar(term, id, self.ext)
                        if avatar is not None and not self.exists(avatar_file):
                            with open(avatar_file, 'wb') as f:
                                f.write(avatar)
                        else:
                            self.logger.info('ignoring default avatar for {0}'
                                             .format(id))
                with open(ids_map_file, 'w+') as f:
                    ids_map = json.dumps(ids_map, indent=4)
                    f.write(ids_map)
                self.logger.info('updated ids map: {0}'.format(ids_map_file))
        return TaskResult.OK
