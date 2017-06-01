
import json
from task import Task, TaskResult


class AvatarLoader(Task):

    ext = 'jpg'

    def __init__(self, config, logger, client):
        super().__init__(config, logger)
        self.client = client

    def routine(self):
        for term in self.config.search_terms:            
            for p in range(1, self.config.search_results.page_count + 1):
                _, _, _, ids_page_file = self.paths_for(term, p) 
                with open(ids_page_file, 'r') as f:
                    ids_map_cnt = f.read()
                    ids_map = json.loads(ids_map_cnt)
                filtered_map = {id: stat for id, stat in ids_map.items() 
                                if not stat['s']}
                for id, stat in filtered_map.items():
                    try:                        
                        avatar, url = self.client.avatar(id)                        
                    except Exception:
                        raise
                    else:
                        stat['s'], stat['url'] = 1, url
                        if avatar is not None:
                            avatar_path = self.path_for_avatar(term, id, self.ext)
                            with open(avatar_path, 'wb') as f:
                                f.write(avatar)
                with open(ids_page_file, 'w+') as f:
                    ids_map_str = json.dumps(ids_map, indent=4)
                    f.write(ids_map_str)
        return TaskResult.OK
