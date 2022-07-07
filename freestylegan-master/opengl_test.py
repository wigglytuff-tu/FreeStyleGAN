from graphics_utils.image_io import *
from graphics_utils.ogl_ops import *

renderData = EasyDict()


def getViewerDir():
    return os.path.dirname(os.path.realpath(__file__))


def render():
    renderData.displayOp.render(renderData.tex, toScreen=True)


def testGLFW(image, res, window):
    fbo = createFramebuffer()
    renderData.displayOp = TexturedScreenQuadOP(fbo, res)
    renderData.tex = uploadImage(image)
    render()
    glfw.swap_buffers(window)

    while not glfw.window_should_close(window):
        glfw.poll_events()
    glfw.terminate()


def main():
    displayRes = (1, 1)

    if glfw_ok:  # feedback from glfw initialization
        img = loadImage("./figures/teaser.jpg")
        displayRes = tuple([int(r / 3) for r in img.shape[0:2][::-1]])
        img = cv2.resize(img, dsize=displayRes, interpolation=cv2.INTER_AREA)
    window = oglInit(displayRes, "OpenGL Test")

    if window is None:
        print("OpenGL initialized using EGL headless mode...")
    else:
        print("OpenGL initialized using GLFW...")
        testGLFW(img, displayRes, window)
    print("OpenGL test passed.")


if __name__ == "__main__":
    main()
    print("=== TERMINATED ===")