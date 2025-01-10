from google_images_search import GoogleImagesSearch

# Replace these with your Google API key and CX
gis = GoogleImagesSearch('AIzaSyBtuwJMDJ3KTG8zRft3CIA6besO0e5AV8o', '70e44bf2773ef4d15')

# Define search parameters
search_params_bird = {
    'q': 'bird or birdsfeeding bird drinking',
    'num': 85,
    'safe': 'high',
    'fileType': 'jpg|png',
    'imgType': 'photo',
    'imgSize': 'medium',
}

# Execute the search for birds
search_results_bird = gis.search(search_params_bird, path_to_dir='/Users/kailashkumar/Documents/CODE/bird-detector/data/bird')

# Check if search results exist before attempting to download
if search_results_bird:
    for image in search_results_bird:
        gis.download(url=image.url, path_to_dir='/Users/kailashkumar/Documents/CODE/bird-detector/data/bird')
else:
    print("No search results for birds.")

# Define search parameters for non-birds
search_params_non_bird = {
    'q': 'random things',
    'num': 500,
    'safe': 'high',
    'fileType': 'jpg|png',
    'imgType': 'photo',
    'imgSize': 'medium',
}

# Execute the search for non-birds
search_results_non_bird = gis.search(search_params_non_bird, path_to_dir='/Users/kailashkumar/Documents/CODE/bird-detector/data/non-bird')

# Check if search results exist before attempting to download
if search_results_non_bird:
    for image in search_results_non_bird:
        gis.download(url=image.url, path_to_dir='/Users/kailashkumar/Documents/CODE/bird-detector/data/non-bird')
else:
    print("No search results for non-birds.")