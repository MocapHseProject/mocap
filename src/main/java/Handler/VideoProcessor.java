package main.java.Handler;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;

public class VideoProcessor {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.out.println("Version: " + Core.VERSION);
    }

    private final VideoCapture cap;
    private final boolean isEnd;

    /**
     * Video Processor Constructor for mp4
     *
     * @param path path to mp4 video format
     */
    VideoProcessor(String path) {
        this.cap = new VideoCapture(path);
        this.isEnd = false;
    }

    /**
     * Video Processor Constructor for front camera
     */
    VideoProcessor() {
        this.cap = new VideoCapture(0);
        this.isEnd = false;
    }

    /**
     * Video Processor Constructor for connected devices
     */
    VideoProcessor(int type) {
        this.cap = new VideoCapture(type);
        this.isEnd = false;
    }

    /**
     * Checking for end of video stream function.
     */
    public Boolean isEnd() {
        return isEnd;
    }

    /**
     * Function returning the next frame of a video stream
     *
     * @return pixels buffer and image dimensions
     */

    public Frame next() {
        if (isEnd) {
            return null;
        }
        if (!cap.grab()) {
            cap.release();
            return null;
        }
        Mat frame = new Mat();
        cap.read(frame);


        ByteBuffer pixels = ByteBuffer.allocate(frame.height() * frame.width() * 3);

        pixels.rewind();
        for (int y = 0; y < frame.height(); ++y) {
            for (int x = 0; x < frame.width(); ++x) {
                double[] colors = frame.get(y, x);
                pixels.put((byte) colors[2]);
                pixels.put((byte) colors[1]);
                pixels.put((byte) colors[0]);
            }
        }
        return new Frame(pixels, frame.height(), frame.width());
    }

    /**
     * VideoProcessor testing
     */

    public static void main(String[] args) {
        VideoProcessor processor = new VideoProcessor();
        Frame frame = processor.next();

        BufferedImage image = new BufferedImage(frame.getWeight(), frame.getHeight(), BufferedImage.TYPE_INT_RGB);

        frame.getBytes().rewind();

        for (int y = 0; y < frame.getHeight(); y++) {
            for (int x = 0; x < frame.getWeight(); x++) {
                int r = frame.getBytes().get() & 0xFF;
                int g = frame.getBytes().get() & 0xFF;
                int b = frame.getBytes().get() & 0xFF;

                int rgb = (r << 16) | (g << 8) | b;
                image.setRGB(x, y, rgb);
            }
        }

        try {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ImageIO.write(image, "png", baos);

            File imageFile = new File("output.png");
            ImageIO.write(image, "png", imageFile);
            System.out.println("PNG изображение успешно создано");
        } catch (IOException e) {
            System.out.println("Ошибка при создании изображения: " + e.getMessage());
        }
    }
}
