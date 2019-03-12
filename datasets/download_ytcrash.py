import requests, os

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

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

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_and_extract(file_id, dst, final_dst):
    print("downloading {}...".format(dst))
    download_file_from_google_drive(file_id, dst)
    print("download completed!")

    print("extracting...")
    mkdir(final_dst)
    mkdir("./datasets/temp")
    os.system("tar -xf {} --strip-components=1 -C ./datasets/temp".format(dst))
    os.system("mv ./datasets/temp/* {}".format(final_dst))
    os.system("rm -r ./datasets/temp")
    print("extraction completed!")

    os.system("rm {}".format(dst))
    print("zip file removed")

if __name__ == "__main__":
    file_id = str("1mC1nlkLa6ffU09ChbQLK7d7sN_B-rMxP")
    dst = str("./datasets/YouTubeCrash_train_accident_images.tar.gz")
    final_dst = str("./datasets/YouTubeCrash/train/accident")
    download_and_extract(file_id, dst, final_dst)

    file_id = str("1lkeswrHasRF1tH6selq5kp-JTbCvfkjS")
    dst = str("./datasets/YouTubeCrash_train_accident_labels.tar.gz")
    final_dst = str("./datasets/YouTubeCrash/train/accident")
    download_and_extract(file_id, dst, final_dst)

    file_id = str("1Ofpe5ZKlf3pWix8Ho3Zm-7ZpxEIH_8qL")
    dst = str("./datasets/YouTubeCrash_train_nonaccident_images.tar.gz")
    final_dst = str("./datasets/YouTubeCrash/train/nonaccident")
    download_and_extract(file_id, dst, final_dst)

    file_id = str("1q-VaEHBgSCb_ugob7TPVZrqQX_9CtdUc")
    dst = str("./datasets/YouTubeCrash_train_nonaccident_labels.tar.gz")
    final_dst = str("./datasets/YouTubeCrash/train/nonaccident")
    download_and_extract(file_id, dst, final_dst)

    file_id = str("1xGn2O-N6_ONAwWnbjQKSpcTloU6QLapR")
    dst = str("./datasets/YouTubeCrash_test_accident_images.tar.gz")
    final_dst = str("./datasets/YouTubeCrash/test/accident")
    download_and_extract(file_id, dst, final_dst)

    file_id = str("11fdzKTSensNlwQpau0Gi8Ikraf0yhYcQ")
    dst = str("./datasets/YouTubeCrash_test_accident_labels.tar.gz")
    final_dst = str("./datasets/YouTubeCrash/test/accident")
    download_and_extract(file_id, dst, final_dst)

    file_id = str("15VMXowVcbZVVm7aOw2wsgCuzqmUMZanX")
    dst = str("./datasets/YouTubeCrash_test_nonaccident_images.tar.gz")
    final_dst = str("./datasets/YouTubeCrash/test/nonaccident")
    download_and_extract(file_id, dst, final_dst)

    file_id = str("1lLEY3Gj6sWJCDcGLi0RYomeWCtTHXQIt")
    dst = str("./datasets/YouTubeCrash_test_nonaccident_labels.tar.gz")
    final_dst = str("./datasets/YouTubeCrash/test/nonaccident")
    download_and_extract(file_id, dst, final_dst)


