
import json
import http.client
from urllib.parse import urlparse

class FacebookClient(object):
	
    version = '2.9'
    base_url = 'graph.facebook.com'

    def __init__(self, logger, access_token):
        self.logger = logger
        self.access_token = access_token

    def search(self, term, offset, size):
        """
        {
           "data": [
              {
                 "id": "863602833653665"
              }
           ],
           "paging": {
              "next": "https://graph.facebook.com/v2.9/...",
              "previous": "https://graph.facebook.com/v2.9/..."
           }
        }
        """
        url = '/v{0}/search?fields=id&type=user&q={1}&format=json&offset={2}&limit={3}&access_token={4}'.format(
            self.version, term, offset, size, self.access_token)
        self.logger.debug('search request - {0}'.format(url[:100]))
        conn = http.client.HTTPSConnection(self.base_url)
        conn.request('GET', url)
        resp = conn.getresponse()
        if resp.status != 200:
            conn.close()
            raise Exception('search failed: {0} - {1}'.format(resp.status, resp.reason))
        resp_data = resp.read()
        conn.close()
        resp_parsed = json.loads(resp_data)        
        ids = [d['id'] for d in resp_parsed['data']]
        return ids

    def avatar(self, id):
        """
        {
           "picture": {
              "data": {
                 "is_silhouette": false,
                 "url": "https://scontent.xx.fbcdn.net/v/t1.0-1/p100x100/..."
              }
           },
           "id": "10203954828725814"
        }
        """
        url = '/v{0}/{1}?fields=id,picture.type(large)&access_token={2}'.format(
            self.version, id, self.access_token)
        self.logger.debug('profile request - {0}'.format(url[:100]))
        conn = http.client.HTTPSConnection(self.base_url)
        conn.request('GET', url)
        resp = conn.getresponse()
        if resp.status != 200:
            conn.close()
            raise Exception('get profile failed: {0} - {1}'.format(resp.status, resp.reason))
        resp_data = resp.read()
        conn.close()
        resp_parsed = json.loads(resp_data)
        url = resp_parsed['picture']['data']['url']  
        if resp_parsed['picture']['data']['is_silhouette']:
            return None, url
        self.logger.debug('avatar request - {0}'.format(url[:100]))        
        url_parsed = urlparse(url)
        url_relative = '{0}?{1}'.format(url_parsed.path, url_parsed.query)
        conn = http.client.HTTPSConnection(url_parsed.netloc)
        conn.request('GET', url_relative)
        resp = conn.getresponse()
        if resp.status != 200:
            conn.close()
            raise Exception('get avatar failed: {0} - {1}'.format(resp.status, resp.reason))
        avatar = resp.read()
        conn.close()
        return avatar, url
