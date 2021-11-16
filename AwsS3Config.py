
class AwsS3Config:
    def __init__(self, accessKey, secretkey):
        self.accessKey = accessKey
        self.secretKey = secretkey
    # http://s3-REGION-.amazonaws.com/BUCKET-NAME/KEY
    def uploadImageToS3Bucket(self):
        pass