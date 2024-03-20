package main.java.Parser.ParserDAE;

import java.nio.FloatBuffer;
import java.util.List;

import main.java.Parser.ParserXML.NodeXML;
import main.java.MyMath.Matrix;

public class SkeletonParser {

    private NodeXML armatureData;


    private Joint loadJointData(NodeXML jointNode, boolean isRoot){
        JointData joint = extractMainJointData(jointNode, isRoot);
        for(NodeXML childNode : jointNode.getChildren("node")){
            joint.addChild(loadJointData(childNode, false));
        }
        return joint;
    }

    private Joint extractMainJointData(NodeXML jointNode, boolean isRoot){
        String nameId = jointNode.parameters.get("id");
        int index = boneOrder.indexOf(nameId);
        String[] matrixData = jointNode.getChild("matrix").data.split(" ");
        Matrix matrix = new Matrix(convertData(matrixData).array());
        jointCount++;
        return new JointData(index, nameId, matrix);
    }

    private List<String> boneOrder;

    private int jointCount = 0;

    public SkeletonLoader(NodeXML visualSceneNode, List<String> boneOrder) {
        this.armatureData = visualSceneNode.getChild("visual_scene").getChildWithAttribute("node", "id", "Armature");
        this.boneOrder = boneOrder;
    }
}
