import ffmpy
import youtube_dl


def download_video(url, output):
    ydl_opts = {
        'format': 'bestvideo',
        'quiet': True,
        'outtmpl': output
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def crop(in_path, out_path, start_time, duration_time):
    """
    Crops part of video
    :param in_path: path of video file
    :param out_path: path of output video file
    :param start_time: e.g. "00:00:00"
    :param duration_time: e.g. "00:00:00"
    :return: True if succesful
    """
    params = ["-v", "quiet", "-y", "-vcodec", "copy", "-acodec", "copy", "-ss", start_time, "-t", duration_time, "-sn"]
    ff = ffmpy.FFmpeg(inputs={in_path: []}, outputs={out_path: params})
    ff.run()
    return True
