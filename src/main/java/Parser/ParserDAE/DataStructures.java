
package main.java.Parser.ParserDAE;
import main.java.MyMath.Matrix;
import main.java.MyMath.Vector;

import java.util.ArrayList;
import java.util.List;

class Model {

    private Skeleton skeleton;
    public Mesh mesh;

    public Model(Mesh mesh, Skeleton skeleton){
        this.skeleton = skeleton;
        this.mesh = mesh;
    }

}


class Joint {


    public Joint(int index, String id, Matrix transformation) {
        this.id = id;
        this.transformation = transformation;
    }

    public String id;
    public Matrix transformation;
    public List<Joint> children = new ArrayList<Joint>();


}

class Transformation {

    public public String id;
    public public Matrix transform;

    public Transformation(String id, Matrix transform) {
        this.id = id;
        this.transform = transform;
    }
}

class Vertex {



    public SkinVertex weights;

    public Vertex(int index,Vector position, SkinVertex weights){
        this.index = index;
        this.weights = weights;
        this.position = position;
        this.length = position.length();
    }

    public Vector position;
    public int index;
    public float length;
    public Vector avgT = new Vector(0, 0, 0);
    public List<Vector> tangents = new ArrayList<Vector>();

    public void averageTangents(){
        for(Vector tng : tangents){
            Vector.vectorAddition(avgT, tng);
        }
        avgT.normalize();
    }

}


class SkinVertex {

    public void jointCompose(int jointId, float weight){
        for(int i = 0; i < weights.size(); i++){
            if(weight > weights.get(i)){
                jointIds.add(i, jointId);
                weights.add(i, weight);
                return;
            }
        }
        jointIds.add(jointId);
        weights.add(weight);
    }



    public List<Integer> jointIds = new ArrayList<Integer>();
    public List<Float> weights = new ArrayList<Float>();



    private void updateWeights(float[] startWeights, float total){
        weights.clear();
        for(int i = 0 ;i < startWeights.length; i++){
            weights.add(Math.min(startWeights[i]/total, 1));
        }
    }

    private void filterJoints(int max){
        while(jointIds.size() > max){
            jointIds.remove(jointIds.size()-1);
        }
    }

    private void fillEmptyWeights(int max){
        while(jointIds.size() < max){
            jointIds.add(0);
            weights.add(0f);
        }
    }
}

class KeyFrame {

    public float time;
    public List<Transformation> jointTransforms = new ArrayList<Transformation>();

    public KeyFrame(float time){
        this.time = time;
    }


}

class Mesh {


    public Mesh(float[] vertices, float[] uv, float[] normals, int[] indices,
                int[] jointIds, float[] weights) {
        this.vertices = vertices;
        this.uv = uv;
        this.normals = normals;
        this.indices = indices;
        this.jointIds = jointIds;
        this.weights = weights;
    }

    public float[] vertices;
    public float[] uv;
    public int[] jointIds;
    public float[] weights;
    public float[] normals;
    public int[] indices;

}


class Skeleton {

    public Skeleton(int connectionCnt, Joint rootJoint){
        this.connectionCnt = connectionCnt;
        this.rootJoint = rootJoint;
    }
    public int connectionCnt;
    public Joint rootJoint;


}

class Skinning {



    public Skinning(List<String> jointsIns, List<SkinVertex> skinVerticies){
        this.jointsIns = jointsIns;
        this.skinVerticies = skinVerticies;
    }

    public List<String> jointsIns;
    public List<SkinVertex> skinVerticies;
}

class Animation {

    public float secs;

    public Animation(float secs, KeyFrame[] keyFrames) {
        this.secs = secs;
        this.keyFrames = keyFrames;
    }
    public KeyFrame[] keyFrames;
}
