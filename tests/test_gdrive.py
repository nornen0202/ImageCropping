from crop_datasets.download.gdrive import gdrive_url


def test_gdrive_url_builder():
    assert gdrive_url("abc123") == "https://drive.google.com/uc?id=abc123"
