package main.java.Handler;

import java.util.ArrayList;
import java.util.List;

public class JointHandler {

    /**
     * class Joints contains coordinates of joints in 2D space
     */
    private static class Joints {
        public List<Integer> cords;

        Joints() {
            cords = new ArrayList<>(6);
        }
    }

    private Joints previousJoint;
    private Boolean isEnd;
    private final VideoProcessor processor;
    private final List<Byte> color1;
    private final List<Byte> color2;
    private final List<Byte> color3;
    private int permissibleError;
    private final int standardPermissibleError;

    /**
     * Video Processor Constructor for mp4
     *
     * @param path path to mp4 video format
     */
    JointHandler(String path) {
        this.processor = new VideoProcessor(path);
        isEnd = false;
        color1 = List.of((byte) 0, (byte) 0, (byte) 255);
        color2 = List.of((byte) 0, (byte) 255, (byte) 0);
        color3 = List.of((byte) 255, (byte) 0, (byte) 255);
        permissibleError = 15;
        standardPermissibleError = 15;
    }

    /**
     * Video Processor Constructor for connected devices
     * use param 0 for front camera
     */
    JointHandler(int type) {
        this.processor = new VideoProcessor(type);
        isEnd = false;
        color1 = List.of((byte) 0, (byte) 0, (byte) 255);
        color2 = List.of((byte) 0, (byte) 255, (byte) 0);
        color3 = List.of((byte) 255, (byte) 0, (byte) 255);
        permissibleError = 15;
        standardPermissibleError = 15;
    }

    /**
     * Checking for end of video stream function.
     */
    Boolean isEnd() {
        return isEnd;
    }

    /**
     * Function handles the following frame of video
     *
     * @return coordinates of joints in 2D space
     * if value is null, then either the video stream has ended,
     * or there is a problem with image processing
     */
    List<Integer> Next() {
        if (processor.isEnd()) {
            isEnd = true;
            return null;
        }

        Frame frame = processor.next();
        int count = 0;
        while (previousJoint == null) {
            if (count++ > 2) {
                return null;
            }
            previousJoint = new Joints();
            int errorCode;
            errorCode = searchWithoutPrevious(frame, color1);
            if (errorCode != 0) {
                if (errorCode == 1) {
                    permissibleError += 5;
                    previousJoint = null;
                    continue;
                }
                if (errorCode == 2) {
                    permissibleError -= 5;
                    previousJoint = null;
                    continue;
                }
            }
            errorCode = searchWithoutPrevious(frame, color2);
            if (errorCode != 0) {
                if (errorCode == 1) {
                    permissibleError += 5;
                    previousJoint = null;
                    continue;
                }
                if (errorCode == 2) {
                    permissibleError -= 5;
                    previousJoint = null;
                    continue;
                }
            }
            errorCode = searchWithoutPrevious(frame, color3);
            if (errorCode != 0) {
                if (errorCode == 1) {
                    permissibleError += 5;
                    previousJoint = null;
                    continue;
                }
                if (errorCode == 2) {
                    permissibleError -= 5;
                    previousJoint = null;
                    continue;
                }
            }
            permissibleError = standardPermissibleError;
            return previousJoint.cords;
        }
        return searchWithPrevious(frame);
    }

    /**
     * Function checks pixel color
     *
     * @param color pattern
     * @param frame color of pixel
     */
    Boolean checkByte(Byte color, Byte frame) {
        return Math.abs(color - frame) < permissibleError;
    }

    /**
     * Function checks for image out of bounds
     *
     * @param w     width coordinate
     * @param h     height coordinate
     * @param frame image
     */
    Boolean checkCords(int w, int h, Frame frame) {
        if (w >= frame.getWeight() || w < 0) return false;
        return h < frame.getHeight() && h >= 0;
    }

    /**
     * Function determines the position of the joints in primary processing
     *
     * @param frame image
     * @param color sample search
     * @return error code
     * 0 - success
     * 1 - object found too large
     * 2 - object not found
     */

    int searchWithoutPrevious(Frame frame, List<Byte> color) {
        List<Boolean> table = new ArrayList<>(frame.getWeight() * frame.getWeight());
        frame.getBytes().flip();
        int minW = Integer.MAX_VALUE, maxW = 0, minH = Integer.MAX_VALUE, maxH = 0;
        for (int i = 0; i < frame.getHeight(); i++) {
            for (int j = 0; j < frame.getWeight(); j++) {
                boolean point = true;
                for (int c = 0; c < 3; c++) {
                    point &= checkByte(color.get(c), frame.getBytes().get());
                }
                table.add(point);
                if (point) {
                    minW = Math.min(minW, j);
                    minH = Math.min(minH, i);
                    maxW = Math.max(maxW, j);
                    maxH = Math.max(maxH, i);
                }
            }
        }
        if (maxW - minW > 100 || maxH - minH > 100) {
            return 1;
        }
        if (maxW - minW <= 7 || maxH - minH <= 7) {
            return 2;
        }

        previousJoint.cords.add((minW + maxW) / 2);
        previousJoint.cords.add((minH + maxH) / 2);
        return 0;
    }

    /**
     * class Limits stores the boundaries of the detected object
     */
    private static class Limits {
        int minH, maxH, minW, maxW;

        Limits() {
            minW = Integer.MAX_VALUE;
            maxW = 0;
            minH = Integer.MAX_VALUE;
            maxH = 0;
        }

        /**
         * Function updates object boundaries
         */
        void update(int i, int j) {
            minW = Math.min(minW, j);
            minH = Math.min(minH, i);
            maxW = Math.max(maxW, j);
            maxH = Math.max(maxH, i);
        }
    }

    /**
     * Recursive image traversal for first color
     *
     * @param cordW width coordinate
     * @param cordH height coordinate
     * @param frame image
     * @param limit object boundaries
     * @param used  pixels passed
     */
    void recSearchColor1(int cordW, int cordH, Frame frame, Limits limit, List<Boolean> used) {
        if (!checkCords(cordW, cordH, frame)) {
            return;
        }
        if (used.get(cordW + cordH * frame.getWeight())) {
            return;
        }
        used.set(cordW + cordH * frame.getWeight(), true);
        boolean point = true;
        for (int c = 0; c < 3; c++) {
            point &= checkByte(color1.get(c),
                    frame.getBytes().get(
                            cordH * frame.getWeight() * 3 + cordW * 3 + c
                    ));
        }
        if (!point) {
            return;
        }
        limit.update(cordH, cordW);
        recSearchColor1(
                cordW + 1, cordH,
                frame, limit, used);
        recSearchColor1(
                cordW, cordH + 1,
                frame, limit, used);
        recSearchColor1(
                cordW - 1, cordH,
                frame, limit, used);
        recSearchColor1(
                cordW, cordH - 1,
                frame, limit, used);
    }

    /**
     * Recursive image traversal for second color
     *
     * @param cordW width coordinate
     * @param cordH height coordinate
     * @param frame image
     * @param limit object boundaries
     * @param used  pixels passed
     */
    void recSearchColor2(int cordW, int cordH, Frame frame, Limits limit, List<Boolean> used) {
        if (!checkCords(cordW, cordH, frame)) {
            return;
        }
        if (used.get(cordW + cordH * frame.getWeight())) {
            return;
        }
        used.set(cordW + cordH * frame.getWeight(), true);
        boolean point = true;
        for (int c = 0; c < 3; c++) {
            point &= checkByte(color2.get(c),
                    frame.getBytes().get(
                            cordH * frame.getWeight() * 3 + cordW * 3 + c
                    ));
        }
        if (!point) {
            return;
        }
        limit.update(cordH, cordW);
        recSearchColor2(
                cordW + 1, cordH,
                frame, limit, used);
        recSearchColor2(
                cordW, cordH + 1,
                frame, limit, used);
        recSearchColor2(
                cordW - 1, cordH,
                frame, limit, used);
        recSearchColor2(
                cordW, cordH - 1,
                frame, limit, used);
    }

    /**
     * recursive image traversal for third color
     *
     * @param cordW width coordinate
     * @param cordH height coordinate
     * @param frame image
     * @param limit object boundaries
     * @param used  pixels passed
     */
    void recSearchColor3(int cordW, int cordH, Frame frame, Limits limit, List<Boolean> used) {
        if (!checkCords(cordW, cordH, frame)) {
            return;
        }
        if (used.get(cordW + cordH * frame.getWeight())) {
            return;
        }
        used.set(cordW + cordH * frame.getWeight(), true);
        boolean point = true;
        for (int c = 0; c < 3; c++) {
            point &= checkByte(color3.get(c),
                    frame.getBytes().get(
                            cordH * frame.getWeight() * 3 + cordW * 3 + c
                    ));
        }
        if (!point) {
            return;
        }
        limit.update(cordH, cordW);
        recSearchColor3(
                cordW + 1, cordH,
                frame, limit, used);
        recSearchColor3(
                cordW, cordH + 1,
                frame, limit, used);
        recSearchColor3(
                cordW - 1, cordH,
                frame, limit, used);
        recSearchColor3(
                cordW, cordH - 1,
                frame, limit, used);
    }

    /**
     * Function for determining the position of joints
     * based on data about the previous position
     *
     * @param frame image
     * @return coordinates of joints in 2D space
     */
    List<Integer> searchWithPrevious(Frame frame) {
        List<Boolean> used = new ArrayList<>(frame.getWeight() * frame.getHeight());
        int sizePicture = frame.getWeight() * frame.getHeight();
        for (int i = 0; i < sizePicture; i++) {
            used.add(false);
        }
        List<Integer> cords = new ArrayList<>(6);
        int count = 0;
        while (count++ < 2) {
            Limits limit1 = new Limits();
            recSearchColor1(
                    previousJoint.cords.get(0),
                    previousJoint.cords.get(1),
                    frame,
                    limit1,
                    used);
            if (limit1.maxW - limit1.minW > 100 || limit1.maxH - limit1.minH > 100) {
                permissibleError += 5;
                continue;
            }
            if (limit1.maxW - limit1.minW < 7 || limit1.maxH - limit1.minH < 7) {
                permissibleError -= 5;
                continue;
            }
            cords.add((limit1.minW + limit1.maxW) / 2);
            cords.add((limit1.minH + limit1.maxH) / 2);
            break;
        }
        if (count >= 2) {
            return null;
        }

        permissibleError = standardPermissibleError;
        count = 0;
        while (count++ < 2) {
            Limits limit2 = new Limits();
            recSearchColor2(
                    previousJoint.cords.get(2),
                    previousJoint.cords.get(3),
                    frame,
                    limit2,
                    used);
            if (limit2.maxW - limit2.minW > 100 || limit2.maxH - limit2.minH > 100) {
                permissibleError += 5;
                continue;
            }
            if (limit2.maxW - limit2.minW < 9 || limit2.maxH - limit2.minH < 9) {
                permissibleError -= 5;
                continue;
            }
            cords.add((limit2.minW + limit2.maxW) / 2);
            cords.add((limit2.minH + limit2.maxH) / 2);
            break;
        }
        if (count >= 2) {
            return null;
        }
        permissibleError = standardPermissibleError;
        count = 0;
        while (count++ < 2) {
            Limits limit3 = new Limits();

            recSearchColor3(
                    previousJoint.cords.get(4),
                    previousJoint.cords.get(5),
                    frame,
                    limit3,
                    used);
            if (limit3.maxW - limit3.minW > 100 || limit3.maxH - limit3.minH > 100) {
                permissibleError += 5;
                continue;
            }
            if (limit3.maxW - limit3.minW < 11 || limit3.maxH - limit3.minH < 11) {
                permissibleError -= 5;
                continue;
            }
            cords.add((limit3.minW + limit3.maxW) / 2);
            cords.add((limit3.minH + limit3.maxH) / 2);
            break;
        }
        permissibleError = standardPermissibleError;
        previousJoint.cords = cords;
        return cords;
    }

    public static void main(String[] args) {


    }
}
