# Download code taken from Code taken from https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039
import requests
#https://drive.google.com/a/andrew.cmu.edu/uc?export=download&confirm=I9ZU&id=0B6VhzidiLvjSa19uYWlLUEkzX3c
def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/a/andrew.cmu.edu/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

file_id = '0B6VhzidiLvjSaER5YkJUdWdPWU0'
destination = './model_bi.bin'
download_file_from_google_drive(file_id, destination) 
