import http.client
import urllib


def send_message(api_token, api_user, message):
    conn = http.client.HTTPSConnection("api.pushover.net:443")
    conn.request("POST", "/1/messages.json",
      urllib.parse.urlencode({
        "token": "{}".format(api_token),
        "user": "{}".format(api_user),
        "message": "{}".format(message),
      }), { "Content-type": "application/x-www-form-urlencoded" })

    conn.getresponse()
