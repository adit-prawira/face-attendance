import os
import boto3
import cv2
import urllib.request
import numpy as np
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv
load_dotenv()

def uploadImageToS3(fileLoc, bucketName, accessKey, secretKey, region):
    s3 = boto3.client("s3", aws_access_key_id=accessKey, aws_secret_access_key=secretKey)
    link = ""
    try:
        fileBaseName = os.path.basename(fileLoc)
        s3.upload_file(fileLoc, bucketName, fileBaseName)
        # http://s3-REGION-.amazonaws.com/BUCKET-NAME/KEY
        link = f"https://{bucketName}.s3.{region}.amazonaws.com/{fileBaseName}"
    except NoCredentialsError:
        print("Credential is not provided")
    return link

def openRequestedImage(url, readFlag=cv2.IMREAD_COLOR):
    res = (urllib.request.urlopen(url)).read()
    image = np.asarray(bytearray(res), dtype="uint8")
    return cv2.imdecode(image, readFlag)

accessKey = os.environ.get("AWS_ACCESS_KEY")
secretKey = os.environ.get("AWS_SECRET_KEY")
region = os.environ.get("AWS_REGION")
bucketName = os.environ.get("AWS_BUCKET_NAME")
s3Link = uploadImageToS3("resources/jack-ma.jpeg", bucketName, accessKey, secretKey, region)

if s3Link:
    cv2.imshow("Image", openRequestedImage(s3Link))
    cv2.waitKey(0)

