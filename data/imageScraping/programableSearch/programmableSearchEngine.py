import requests
import json

movie = input('Enter Movie: ')

# replace with your own CSE ID and API key
cse_id = "f6246be7db995404a"
api_key = "AIzaSyA9H2ecwhOX2RdQx0DXrOATpUUpFwIq4fc"

url = f"https://www.googleapis.com/customsearch/v1?q={movie} movie cover&num=1&start=1&imgSize=huge&searchType=image&key={api_key}&cx={cse_id}"

response = requests.get(url)
response.raise_for_status()



search_results = response.json()
image_url = search_results['items'][0]['link']

search_results_file = json.dumps(search_results)

with open("results.json", "w") as file:
    file.write(search_results_file)



print('Image URL:', image_url)