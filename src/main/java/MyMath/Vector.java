package main.java.MyMath;

import static com.jogamp.opengl.math.FloatUtil.sqrt;

// Vector class
public class Vector {
    public float x; // Vector x coordinate
    public float y; // Vector y coordinate
    public float z; // Vector z coordinate

    /**
     * Vector default constructor function.
     */
    public Vector() {
        x = y = z = 0;
    } // End of 'Vector' function

    /**
     * Vector constructor function.
     *
     * @param value value of 3 vector coordinates (all will be the same as the value)
     */
    public Vector(float value) {
        x = y = z = value;
    } // End of 'Vector' function

    /**
     * Vector constructor function.
     *
     * @param x_ vector's x coordinate
     * @param y_ vector's y coordinate
     * @param z_ vector's z coordinate
     */
    public Vector(float x_, float y_, float z_) {
        x = x_;
        y = y_;
        z = z_;
    } // End of 'Vector' functon

    /**
     * Vector constructor function.
     *
     * @param vector array of float values of vector
     */
    public Vector(Float[] vector) {
        if (vector.length < 3) {
            return;
        }
        x = vector[0];
        y = vector[1];
        z = vector[2];
    } // End of 'Vector' function

    /**
     * Getting vector length function.
     *
     * @return vector length float value
     */
    public float length() {
        return sqrt(x * x + y * y + z * z);
    } // End of 'length' function

    /**
     * Getting normalized version of this vector function.
     *
     * @return normalized version of this vector
     * @apiNote do not changing vector itself
     */
    public Vector getNormalized() {
        float length = length();
        return new Vector(x / length, y / length, z / length);
    } // End of 'getNormalized' function

    /**
     * Normalizing vector function.
     *
     * @return vector itself
     * @apiNote changing this vector
     */
    public Vector normalize() {
        float length = length();
        x /= length;
        y /= length;
        z /= length;

        return this;
    } // End of 'normalize' function

    /**
     * Additing 2 vectors function.
     *
     * @param other vector to add
     * @return this vector
     */
    public Vector add(Vector other) {
        x += other.x;
        y += other.y;
        z += other.z;

        return this;
    } // End of 'add' function

    /**
     * Additing 2 vectors function.
     *
     * @param a first vector
     * @param b second vector
     * @return result of adding these vectors
     */
    public static Vector addition(Vector a, Vector b) {
        return new Vector(a.x + b.x, a.y + b.y, a.z + b.z);
    } // End of 'addition' function

    /**
     * Subtracting 2 vectors function.
     *
     * @param other vector to subtract
     * @return this vector
     */
    public Vector subtract(Vector other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;

        return this;
    } // End of 'subtract' function

    /**
     * Subtracting 2 vectors function.
     *
     * @param a first vector
     * @param b second vector
     * @return result of subtracting 2 vectors
     */
    public static Vector subtraction(Vector a, Vector b) {
        return new Vector(a.x - b.x, a.y - b.y, a.z - b.z);
    } // End of 'subtraction' function

    /**
     * Getting scalar product of 2 vectors function.
     *
     * @param other vector to scalar product
     * @return result of scalar product
     */
    public float scalarProduct(Vector other) {
        return x * other.x + y * other.y + z * other.z;
    } // End of 'scalarProduct' function

    /**
     * Getting scalar product of 2 vectors function.
     *
     * @param a first vector
     * @param b second vector
     * @return result of scalar product
     */
    public static float scalarProduction(Vector a, Vector b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    } // End of 'scalarProduct' function

    /**
     * Getting vector product of 2 vectors function.
     *
     * @param other vector to vector product
     * @return this vector
     */
    public Vector vectorProduct(Vector other) {
        Vector tmp = new Vector(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
        x = tmp.x;
        y = tmp.y;
        z = tmp.z;

        return this;
    } // End of 'vectorProduct' function

    /**
     * Getting vector product of 2 vectors function.
     *
     * @param a first vector
     * @param b second vector
     * @return result of vector product
     */
    public static Vector vectorProduction(Vector a, Vector b) {
        return new Vector(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
    } // End of 'vectorProduct' function
} // End of 'Vector' class
