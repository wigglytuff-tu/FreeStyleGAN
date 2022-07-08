import sys, os
import shutil
import subprocess
import image_io
import collage

#===================================================

def findFrames(path):
    filesPNG = []
    filesJPG = []
    files = []
    dformat = None

    extensions = ["png", "jpg", "jpeg"]

    for ext in extensions:
        files = image_io.findImages(path, "." + ext)
        if len(files) != 0:
            dformat = ext
            break
    
    return files, dformat

#===================================================

def renderVideo(path, reverse=True, numRows=1):

    files, dformat = findFrames(path)

    name = path.replace('\\\\', '/').split("/")
    name = name[-1] if name[-1] != "" else name[-2]

    if len(files) == 0:

        print("Collecting files from subdirectories...")

        subDirs = [os.path.join(path, o) for o in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path , o))]
        if len(subDirs) == 0:
            print("No images found.")
            return        
        subFiles = []

        for d in subDirs:
            f, _ = findFrames(d)
            if len(f) == 0:
                print("No images found.")
                return
            subFiles.append(f)

        lengths = [len(s) for s in subFiles]
        if not len(set(lengths)) <= 1:
            print("Subdirectories contain different numbers of images.")
            return

        # create a collage
        for imgIdx in range(lengths[0]):
            f = []
            for dirIdx in range(len(subFiles)):
                f.append(subFiles[dirIdx][imgIdx])
            comp = collage.createCollage(f, numRows)
            outPath = os.path.join(path, "comp_" + str(imgIdx).zfill(4) + ".png")
            image_io.saveImage(comp, outPath)
    
    files, dformat = findFrames(path)    
    
    print("Creating video...")

    # create temp directory
    tempDir = os.path.join(path, "temp\\")
    assert not os.path.exists(tempDir), "Temp file already exists."
    os.mkdir(tempDir)
    
    # copy all images
    for i, file in enumerate(files):
        shutil.copy2(file, os.path.join(tempDir, str(i).zfill(5) + "." + dformat))
    
    # copy twice but in reverse
    if reverse:
        for i, file in enumerate(reversed(files)):
            shutil.copy2(file, os.path.join(tempDir, str(i + len(files)).zfill(5) + "." + dformat))

    # run ffmpeg to produce video
    command = "ffmpeg -i " + tempDir + "%05d." + dformat + " -c:v libx264 -pix_fmt yuv420p -f mp4 " + path + "\\" + name + ".mp4"
    subprocess.call(command, shell=True)

    # delete temp directory
    for file in os.listdir(tempDir):
        os.remove(os.path.join(tempDir, file))
    os.rmdir(tempDir)

#=======================================================

def main(argv):
    path, reverse, numRows = argv[1], bool(argv[2] == '1'), int(argv[3])
    renderVideo(path, reverse, numRows)

#=======================================================

if __name__ == "__main__":    
    main(sys.argv)
    print("=== TERMINATED ===")