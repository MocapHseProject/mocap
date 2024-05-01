package main.java.Handler;

import java.nio.ByteBuffer;
import java.util.List;


public class Frame {
    private final ByteBuffer bytes;
    private final int height;
    private final int weight;

    /**
     * Constructor class Frame .
     *
     * @param byteBuffer pixel buffer
     * @param weight     image width
     * @param height     image height
     */
    public Frame(ByteBuffer byteBuffer, int height, int weight) {
        this.bytes = byteBuffer;
        this.height = height;
        this.weight = weight;
    }

    public int getWeight() {
        return weight;
    }

    public int getHeight() {
        return height;
    }

    public ByteBuffer getBytes() {
        return bytes;
    }
}
