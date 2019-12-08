from modelarts.session import Session

def cifar10_dowloader():
    session = Session()
    bucket_path="/cv-course-public/coding-1/cifar-10-python.tar.gz"
    session.download_data(bucket_path=bucket_path, path="./data/cifar-10-python.tar.gz")

def cifar100_dowloader():
    session = Session()
    bucket_path="/cv-course-public/coding-1/cifar-100-python.tar.gz"
    session.download_data(bucket_path=bucket_path, path="./data/cifar-100-python.tar.gz")