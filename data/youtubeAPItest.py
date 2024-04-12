import requests

url = "https://www.googleapis.com/youtube/v3/channels"

params = {
    "part": "contentDetails",
    "forUsername": "Northernlion",
    "key": "AIzaSyA9H2ecwhOX2RdQx0DXrOATpUUpFwIq4fc"

}

response = requests.get(url, params=params)

data = response.json()

print(data)

#northernlions youtube id: UC3tNpTOHsTnkmbwztCs30sA

