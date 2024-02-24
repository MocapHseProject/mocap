package main.java.Render;

import com.jogamp.opengl.*;
import com.jogamp.newt.event.*;
import com.jogamp.newt.opengl.GLWindow;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.TreeMap;

// Render class
public class Render implements GLEventListener, KeyListener, MouseListener {
    // Button's state enum
    public enum ButtonState {
        CLICKED,
        PRESSED,
        RELEASED;
    } // End of 'ButtonState' enum

    // Mouse handler class
    public class Mouse {
        int x, y;                        // Mouse cursor screen coordinates
        boolean isOnWindow = false;      // Is mouse on rendering window now
        ButtonState[] mouseButtonStates; // States of mouse's buttons

        /**
         * Mouse class default constructor function.
         */
        Mouse() {
            mouseButtonStates = new ButtonState[6]; // Watch MouseEvent structure to recall indexes of mouse buttons
            for (ButtonState state : mouseButtonStates) {
                state = ButtonState.RELEASED;
            }
            x = y = 0;
        } // End of 'Mouse' function
    } // End of 'Mouse' class

    private int width;                                   // Width of rendering window
    private int height;                                  // Height of rendering window
    private float time;                                  // Global time in seconds
    private long startTime;                              // Window creating time in milliseconds
    private Mouse mouse = null;                          // Mouse instance
    private GLWindow window = null;                      // OpenGL window instance
    private ArrayList<Buffer> buffers = null;            // Storage of OpenGL buffers
    private TreeMap<Character, ButtonState> keys = null; // Keyboard buttons

    /**
     * Render initialization function.
     *
     * @throws RuntimeException if OpenGL is not initialized or animator was not created successfully
     */
    private void initialize() throws RuntimeException {
        startTime = System.currentTimeMillis();
        time = 0;
        width = 800;
        height = 600;
        GLProfile.initSingleton();
        if (!GLProfile.isInitialized()) {
            throw new RuntimeException("Error on initializing OpenGL...");
        }
        try {
            GLProfile profile = GLProfile.get(GLProfile.GL3);
            GLCapabilities capabilities = new GLCapabilities(profile);

            window = GLWindow.create(capabilities);
            window.setSize(width, height);
            window.setPointerVisible(true);
            window.setTitle("HSE Project");
            window.setResizable(false);
            window.setVisible(true);

            window.addGLEventListener(this);

            window.addWindowListener(new WindowAdapter() {
                @Override
                public void windowDestroyed(WindowEvent e) {
                    System.exit(1);
                }
            });
        } catch (
                GLException e) {
            System.err.println("Error while initializing OpenGL...");
            throw new RuntimeException(e);
        }
    } // End of 'initialize' function

    /**
     * Render default constructor function.
     */
    public Render() {
        initialize();
    } // End of 'Render' function

    /**
     * Main render class function.
     *
     * @param args command line arguments
     */
    public static void main(String[] args) {
        Render render = new Render();
    } // End of 'main' function

    /**
     * Window initializing function.
     *
     * @param drawable java OpenGL instance
     */
    @Override
    public void init(GLAutoDrawable drawable) {
        gl.glViewport(0, 0, width, height);
        gl.glClearColor(0, 0, 0, 1);
    } // End of 'init' function

    /**
     * Window displaying function.
     *
     * @param drawable java OpenGL instance
     * @apiNote executing every frame
     */
    @Override
    public void display(GLAutoDrawable drawable) {
        time = (float) (System.currentTimeMillis() - startTime) / 1000.0F;

        GL3 gl = drawable.getGL().getGL3();

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT);
        gl.glDisable(gl.GL_BLEND);
        gl.glEnable(gl.GL_DEPTH_TEST);

        gl.glClearColor(0, 0, 0, 1);

        gl.glFinish();
        gl.glDisable(gl.GL_DEPTH_TEST);
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL);
    } // End of 'display' function

    /**
     * Window reshaping function.
     *
     * @param drawable java OpenGL instance
     * @param x        x-axis window position offset
     * @param y        y-axis window position offset
     * @param width_   window new width
     * @param height_  window new height
     */
    @Override
    public void reshape(GLAutoDrawable drawable, int x, int y, int width_, int height_) {
        GL3 gl = drawable.getGL().getGL3();
        width = width_;
        height = height_;
        camera.setProjection(width, height);
        gl.glViewport(x, y, width, height);
    } // End of 'reshape' function

    /**
     * Window disposing function.
     *
     * @param drawable java OpenGL instance
     * @apiNote like a window destructor in C++
     */
    @Override
    public void dispose(GLAutoDrawable drawable) {
        GL3 gl = drawable.getGL().getGL3();
        quit();
    } // End of 'dispose' function

    /**
     * Keyboard event on key typing function.
     *
     * @param e the event to be processed
     */
    @Override
    public void keyTyped(KeyEvent e) {
        keys.put(e.getKeyChar(), ButtonState.CLICKED);
    } // End of 'keyTyped' function

    /**
     * Keyboard event on key pressing function.
     *
     * @param e the event to be processed
     */
    @Override
    public void keyPressed(KeyEvent e) {
        keys.put(e.getKeyChar(), ButtonState.PRESSED);
    } // End of 'keyPressed' function

    /**
     * Keyboard event on key releasing function.
     *
     * @param e the event to be processed
     */
    @Override
    public void keyReleased(KeyEvent e) {
        keys.put(e.getKeyChar(), ButtonState.RELEASED);
    } // End of 'keyReleased' function

    /**
     * Mouse event on mouse buttons clicking function.
     *
     * @param e the event to be processed
     */
    @Override
    public void mouseClicked(MouseEvent e) {
        mouse.mouseButtonStates[e.getButton()] = ButtonState.CLICKED;
    } // End of 'mouseClicked' function

    /**
     * Mouse event on mouse buttons pressing function.
     *
     * @param e the event to be processed
     */
    @Override
    public void mousePressed(MouseEvent e) {
        mouse.mouseButtonStates[e.getButton()] = ButtonState.PRESSED;
    } // End of 'mousePressed' function

    /**
     * Mouse event on mouse buttons releasing function.
     *
     * @param e the event to be processed
     */
    @Override
    public void mouseReleased(MouseEvent e) {
        mouse.mouseButtonStates[e.getButton()] = ButtonState.RELEASED;
    } // End of 'mouseReleased' function

    /**
     * Mouse event on mouse cursor's being on screen function.
     *
     * @param e the event to be processed
     */
    @Override
    public void mouseEntered(MouseEvent e) {
        mouse.isOnWindow = true;
    } // End of 'mouseEntered' function

    /**
     * Mouse event on mouse cursor's not being on screen function.
     *
     * @param e the event to be processed
     */
    @Override
    public void mouseExited(MouseEvent e) {
        mouse.isOnWindow = false;
    } // End of 'mouseExited' function

    /**
     * Quitting render function.
     *
     * @apiNote calling System.exit() synchronously inside the draw, reshape or init callbacks can lead to deadlocks on
     * certain platforms (in particular, X11) because the JAWT's locking routines cause a global AWT lock to be grabbed.
     * Instead, run the exit routine in another thread
     */
    protected void quit() {
        new Thread(new Runnable() {
            public void run() {
                window.destroy();
            }
        }).start();
    } // End of 'quit' function
} // End of 'Render' class
