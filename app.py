import streamlit as st
import swing
import pafy
import os
from pathlib import Path
import openpose
import cv2
st.write(cv2.__version__)
import glob
import os, urllib

path=Path()
path_video=path/'video'
path_video_out=path/'video_output'
path_video.mkdir(exist_ok=True)
path_video_out.mkdir(exist_ok=True)

def main():
    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)

    st.title("Swing Detection")
    source=st.sidebar.radio("Video from:",['Youtube','Upload a mp4 file'])
    swing_det=st.sidebar.radio("Swing Detection Method:",['Onset Detection','Image Classification'])
    device_input=st.sidebar.radio("Device:",['cpu','gpu'])

    if source == 'Youtube':

        youtube_url = st.text_input(
            "Youtube url:", "https://www.youtube.com/watch?v=xpb2Dy-QVHo")
        filename = st.text_input(
            "Filename:", "test")
        download_butt=st.button('Download youtube video')
        if download_butt == True:
            st.write('Downloading Youtube video...')
            youtube_download(youtube_url,filename,path_video)
            st.video(str(path_video/filename)+'.mp4')
            st.write('Youtube video downloaded, click analyse to continue')

        start_analyse = st.button('Analyse')
        
    elif source == 'Upload a mp4 file':
        
        st.set_option('deprecation.showfileUploaderEncoding', False)
        uploaded_file = st.file_uploader(
            "or upload a video file", type=["mp4", "MP4", "avi"])
        filename = st.text_input(
            "Filename:", "test")
        start_analyse = st.button('Analyse')

        if uploaded_file is not None:
            g = io2.BytesIO(uploaded_file.read())  # BytesIO Object
            temporary_location = path_video/(filename_ul+'.mp4')

            with open(temporary_location, 'wb') as out:  # Open temporary file as bytes
                out.write(g.read())  # Read bytes into file

            # close file
            out.close()

    # analyse video
    if start_analyse == True:
        st.write('Analysing...')
        #Swing Detection
        onset_times=swing.onset_detection(path_video/(filename+'.mp4'),cutoff=1000)
        st.write(str(len(onset_times))+ ' swings found.')
        
        #Cut video
        swing.cut_video(filename,onset_times,path_video_out,duration=3)
        
        #create and sort file list
        file_list=glob.glob(str(path_video_out/filename/'*.mp4'),recursive=False)
        file_list.sort(key=lambda x:int(x.split(sep=f'{filename}_')[1].split(sep='.')[0]))
        #st.write(file_list)
        (path_video_out/filename/'output').mkdir(exist_ok=True)
        
        #detect pose for videos
        i=0
        for file_swing in file_list:
            i=i+1
            predict_pose(file_swing,out_path=str(path_video_out/filename/'output')
            , out_name=file_swing.split(sep='/')[-1].split(sep='.')[0],
            dev=device_input)
            st.write(f'Swing {i}: '+file_swing.split(sep='/')[-1].split(sep='.')[0])
            st.video(str(path_video_out/filename/'output'/file_swing.split(sep='/')[-1].split(sep='.')[0])+'.webm')



def youtube_download(url,file,filepath):
    
    video = pafy.new(url)
    clip = video.getbest(preftype="mp4")
    clip.download(filepath)
    os.rename(filepath/clip.filename,filepath/(file+'.mp4'))



def predict_pose(video_path,out_path='video', out_name='output',dev='cpu'):
    net=openpose.load_openpose(dev=dev)
    
    # read the video
    # capture video
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # frame rate of a video
    FPS = cap.get(cv2.CAP_PROP_FPS)
    width_out = 640

    size_out = (width_out, int(width_out*height/width))
    out = cv2.VideoWriter(out_path + '/' + out_name + '.webm',
                          cv2.VideoWriter_fourcc(*'VP90'), FPS, size_out)

    # counter for frame
    cnt = 0

    # counter for detection
    cnt_det = 0

    # Check if video file is opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    while(cap.isOpened()):

        # try:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            print(cnt+1)
            frame = cv2.resize(frame, (width_out, int(
            width_out*height/width)), cv2.INTER_AREA)

            frameClone, personwiseKeypoints=openpose.pose_detect(frame,net,inheight=368)
            out.write(frameClone)
            cnt = cnt+1
            if(cnt == 1):
                out.release()
                break

        # Break the loop
        else:
            out.release()
            break


def download_file(file_path):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                        (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()




EXTERNAL_DEPENDENCIES = {
    "openpose/pose_iter_440000.caffemodel": {
        "url": "https://dl.dropboxusercontent.com/s/chm7rqv10771mzc/pose_iter_440000.caffemodel?dl=0",
        "size": 209274056
    }
}



if __name__ == "__main__":
    main()