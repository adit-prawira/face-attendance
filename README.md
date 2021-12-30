# Face Attendance/Identity Project (Client-side)

The following repository represents the client-side software for the Face Attendance/Identity Project as the overall system of the project is to allow human-computer interaction (HCI) and enables a voice-user interface (VUI). This project utilises OpenCV to perform face recognition and will render the bounding box with the appropriate label for any faces that have been “memorized” by the computer, and will label any face that is not recognized as unknown. 

As a face is unrecognized, this event will trigger the voice assistant that will ask the name of that face and will wait for the user to say the name of that face through the microphone. The voice assistant loves to play defensive as it doesn’t want to make any mistakes and makes the human hate it. Thus, it asks the user for confirmation if the name is right and does the user allow the computer to memorize his or her face. If the user says anything that indicates anything to proceed to the next step, then the voice assistant will send a signal to the face recognition system to get the ROI of the face and send that data to the S3 image uploader.

Finally, the S3 image uploader will convert the array of the face’s ROI into an image, upload it to S3 and return a link to access those images to the client-side. Then the name of the face and the received image URL will be sent to the server and will be stored in the database. As the back response, the server will return the new face data along with its id, name, and its encoded values which will the added to the current name list, and encoded face lists state.
Below is the diagram that pretty much describes how the system interacts with the user and how the client-side communicate with the server-side. (*The same diagram is used in https://github.com/adit-prawira/face_attendance_server)



## How to use the client-side:

To use the client-side software, you must clone the following repo  [*https://github.com/adit-prawira/face_attendance_server*](https://github.com/adit-prawira/face_attendance_server), read the instruction within that repo and run the server in your local machine. If the server is not running, and the client-side is running, it will retry to reconnect to the server every 2 seconds until a connection to the server is made. 

You are required to create a **.env** file that will store your Django secret key and **having an AWS account is a prerequisite** as you create your S3 bucket, access key, and its secret key. Below is the sample of content that should be put inside the **.env** file:

```
AWS_SECRET_KEY=Replace_it_with_your_aws_secret_key
AWS_ACCESS_KEY=Replace_it_with_your_aws_access_key
AWS_REGION=Replace_it_with_your_aws_region
AWS_BUCKET_NAME=Replace_it_with_your_aws_s3_bucket_name
BACKEND_BASE_URL=http://localhost:8000
LAPTOP_CAM_CODE=Replace_it_with_your_laptop_camera_code_usually_0
WEBCAM_CODE=Replace_it_with_your_external_camera_code
```

### Install packages:

Before running the program, you are required to install its dependencies. However, since the virtual environment is utilised for this project to python version consistency please run the following command in your terminal window, the current location of the terminal is located in the current directory.  The command below will activate the virtual environment for this project.

```
source venv/bin/activate
```

After activating the virtual environment, run the following commands to install all dependencies specified in the requirements.txt file.

```
pip install -r requirements.txt
```

### Running the software:

Finally, to run the software you will just simply type the following command in your terminal window

```
python main.py
```

