
Facebook API for Fetching User Avatars
======================================

Access Tokens
-------------

https://developers.facebook.com/tools/explorer/

Templates
---------

https://graph.facebook.com/v2.9/search?fields=id&type=user&q=[query]&format=json&limit=10&offset=0&access_token=[access_teken]

https://graph.facebook.com/v2.9/10203954828725814?fields=picture.type(normal)&access_token=[access_teken]

Example
-------

https://graph.facebook.com/v2.9/search?fields=id&type=user&q=alex&format=json&limit=5&offset=10&access_token=EAACEdEose0cBAGMQ7ul9BcaeWaJOM8HOy5gSbgi8FKCPiZBS6V4B5i5ZCdyhMYZAZC01dLqUXMBevcQQshqRg7E55ZC5b9vZB8xQwKe4EO06T4ud7DL33gPzrIa2OBnCVmy9zRlHSrGepOibFGnxknxNWSB4GfMyZA3s0LQ5QNTB1oUnOryKJhZAhrypHfge1foZD

{
   "data": [
      {
         "id": "863602833653665"
      },
      {
         "id": "10152501608440329"
      },
      {
         "id": "229696210868603"
      },
      {
         "id": "1826682164241378"
      },
      {
         "id": "151195985337118"
      }
   ],
   "paging": {
      "next": "https://graph.facebook.com/v2.9/search?fields=id&limit=5&offset=15&type=user&q=alex&format=json&access_token=EAACEdEose0cBAGMQ7ul9BcaeWaJOM8HOy5gSbgi8FKCPiZBS6V4B5i5ZCdyhMYZAZC01dLqUXMBevcQQshqRg7E55ZC5b9vZB8xQwKe4EO06T4ud7DL33gPzrIa2OBnCVmy9zRlHSrGepOibFGnxknxNWSB4GfMyZA3s0LQ5QNTB1oUnOryKJhZAhrypHfge1foZD&__after_id=enc_AdApfU80t9SGnFRUruLlavztlvFN3kUJSIQueIpyOZCWoZCuHZBXl4Jd03lQTUOUOWqHZC9gt4v48KunHwVZA0oIrvZANC",
      "previous": "https://graph.facebook.com/v2.9/search?fields=id&limit=5&offset=5&type=user&q=alex&format=json&access_token=EAACEdEose0cBAGMQ7ul9BcaeWaJOM8HOy5gSbgi8FKCPiZBS6V4B5i5ZCdyhMYZAZC01dLqUXMBevcQQshqRg7E55ZC5b9vZB8xQwKe4EO06T4ud7DL33gPzrIa2OBnCVmy9zRlHSrGepOibFGnxknxNWSB4GfMyZA3s0LQ5QNTB1oUnOryKJhZAhrypHfge1foZD&__before_id=enc_AdBZBrZBWqsGgOdniZAZAGUa9bqZCdYYlkNE4khvZASKx9DdLbZAwEAka4zibpOsV1ZBxokcALsOPZADvwzAAGirl39wK3gil"
   }
}

https://graph.facebook.com/v2.9/10203954828725814?fields=picture.type(normal)&access_token=EAACEdEose0cBAGMQ7ul9BcaeWaJOM8HOy5gSbgi8FKCPiZBS6V4B5i5ZCdyhMYZAZC01dLqUXMBevcQQshqRg7E55ZC5b9vZB8xQwKe4EO06T4ud7DL33gPzrIa2OBnCVmy9zRlHSrGepOibFGnxknxNWSB4GfMyZA3s0LQ5QNTB1oUnOryKJhZAhrypHfge1foZD

{
   "picture": {
      "data": {
         "is_silhouette": false,
         "url": "https://scontent.xx.fbcdn.net/v/t1.0-1/p100x100/10403120_10206469630954298_5733102400358788670_n.jpg?oh=9625669590d8a79eb52e7bd591737672&oe=59B63696"
      }
   },
   "id": "10203954828725814"
}

