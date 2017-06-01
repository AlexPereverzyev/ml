
import json
from task import Task, TaskResult


class AvatarSearcher(Task):
    def __init__(self, config, logger, client):
        super().__init__(config, logger)
        self.client = client

    def routine(self):
        for term in self.config.search_terms:            
            offset = self.config.search_results.offset
            size =  self.config.search_results.page_size
            for p in range(1, self.config.search_results.page_count + 1):
                term_dir, _, _, ids_page_file = self.paths_for(term, p)
                try:
                    ids = self.client.search(term, offset, size)
                except Exception:
                    raise
                ids_map = json.dumps({id: self.default_map for id in ids}, indent=4)
                with open(ids_page_file, 'w+') as f:
                    f.write(ids_map)
                offset += size
        return TaskResult.OK
